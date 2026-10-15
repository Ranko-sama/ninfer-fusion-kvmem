// FP16 recurrent-state storage against FP32 over a long decode, through the public
// gated_delta_net Op. Both chains see identical inputs; the FP16 chain rounds its stored state
// once per call. Reports the output error per step (relative L2 over all heads) and fails if it
// exceeds 2e-2 -- five bf16 rounding steps -- anywhere, or if it trends upward (the last quarter's
// mean above twice the second quarter's).
// Non-finite outputs and error statistics are always rejected.
//
// Inputs mimic the 27B's layer statistics loosely: 16 query/key heads and 48 value heads,
// normalized q/k, per-head decays from short (alpha 0.9) to near-lossless memory (alpha 0.9999),
// and beta in (0, 1).

#include "ninfer/ops/gated_delta_net.h"
#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/small_t_oracle.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
namespace oracle = ninfer::test::small_t_oracle;

constexpr int kQk    = 16;
constexpr int kHv    = 48;
constexpr int kDim   = 128;
constexpr int kSteps = 4096;

double relative_output_error(std::span<const std::uint16_t> reference,
                             std::span<const std::uint16_t> actual, int call) {
    if (reference.empty() || reference.size() != actual.size()) {
        throw std::invalid_argument("invalid output comparison shape");
    }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double a = oracle::bf16_to_f32(reference[i]);
        const double b = oracle::bf16_to_f32(actual[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            throw std::runtime_error("non-finite " +
                std::string(!std::isfinite(a) ? "FP32-state" : "FP16-state") +
                " output at call " + std::to_string(call) + " element " + std::to_string(i));
        }
        num += (a - b) * (a - b);
        den += a * a;
    }
    const double error = std::sqrt(num / std::max(den, 1e-30));
    if (!std::isfinite(error)) {
        throw std::runtime_error("non-finite relative output error at call " +
                                 std::to_string(call));
    }
    return error;
}

struct ErrorSummary {
    double worst;
    double q1, q2, q3, q4;
    bool passed() const { return worst <= 2e-2 && q4 <= 2.0 * q2; }
};

ErrorSummary summarize_errors(std::span<const double> errors) {
    if (errors.size() < 4) { throw std::invalid_argument("four error quarters are required"); }
    double worst = 0.0;
    for (std::size_t call = 0; call < errors.size(); ++call) {
        if (!std::isfinite(errors[call])) {
            throw std::runtime_error("non-finite error history at call " + std::to_string(call));
        }
        worst = std::max(worst, errors[call]);
    }
    const auto mean = [&](std::size_t begin, std::size_t end) {
        double sum = 0.0;
        for (std::size_t i = begin; i < end; ++i) { sum += errors[i]; }
        return sum / (end - begin);
    };
    const std::size_t calls = errors.size();
    const ErrorSummary summary{worst, mean(0, calls / 4), mean(calls / 4, calls / 2),
                               mean(calls / 2, 3 * calls / 4), mean(3 * calls / 4, calls)};
    if (!std::isfinite(summary.worst) || !std::isfinite(summary.q1) ||
        !std::isfinite(summary.q2) || !std::isfinite(summary.q3) || !std::isfinite(summary.q4)) {
        throw std::runtime_error("non-finite error summary");
    }
    return summary;
}

int finite_gate_regression() {
    int failures = 0, cases = 0;
    const auto expect = [&](const std::string& name, bool actual, bool expected) {
        ++cases;
        if (actual != expected) {
            std::cerr << "FAIL finite gate " << name << ": expected " << expected
                      << ", got " << actual << '\n';
            ++failures;
        }
    };
    const auto accepts = [](std::span<const double> errors) {
        try { return summarize_errors(errors).passed(); }
        catch (const std::exception&) { return false; }
    };
    const std::vector<double> clean(16, 0.001);
    expect("finite errors", accepts(clean), true);
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
        for (int quarter = 0; quarter < 4; ++quarter) {
            auto errors = clean;
            errors[quarter * 4] = bad; // The remaining calls stay finite.
            expect("non-finite quarter " + std::to_string(quarter + 1) + " value " +
                       std::to_string(bad), accepts(errors), false);
        }
    }
    expect("empty history", accepts({}), false);
    expect("too few quarters", accepts(std::vector<double>(3, 0.001)), false);
    expect("maximum error boundary", accepts(std::vector<double>(16, 0.02)), true);
    expect("maximum error exceeded", accepts(std::vector<double>(16, 0.021)), false);
    auto trend = clean;
    std::fill(trend.begin() + 12, trend.end(), 0.002);
    expect("trend boundary", accepts(trend), true);
    std::fill(trend.begin() + 12, trend.end(), 0.0021);
    expect("trend exceeded", accepts(trend), false);

    const auto output_accepts = [&](const std::vector<std::uint16_t>& reference,
                                    const std::vector<std::uint16_t>& actual) {
        try {
            // Exercise the exact comparison and history gate used by the GPU test.
            return accepts(std::vector<double>(16, relative_output_error(reference, actual, 0)));
        } catch (const std::exception&) { return false; }
    };
    expect("identical outputs", output_accepts({0x3f80, 0xbf80}, {0x3f80, 0xbf80}), true);
    expect("zero outputs", output_accepts({0, 0}, {0, 0}), true);
    expect("small finite error", output_accepts({0x3f80}, {0x3f81}), true);
    expect("zero reference floor", output_accepts({0}, {0x3f80}), false);
    // BF16 encodings are exact fixtures, independent of the comparison implementation.
    for (const std::uint16_t bad : {std::uint16_t{0x7fc0}, std::uint16_t{0x7f80},
                                    std::uint16_t{0xff80}}) {
        for (const bool reference_bad : {false, true}) {
            const std::vector<std::uint16_t> reference{reference_bad ? bad : std::uint16_t{0x3f80}};
            const std::vector<std::uint16_t> actual{reference_bad ? std::uint16_t{0x3f80} : bad};
            expect(std::string(reference_bad ? "FP32" : "FP16") + " chain non-finite " +
                       std::to_string(bad), output_accepts(reference, actual), false);
        }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " FP16 finite gate: " << cases
              << " cases, " << failures << " failures\n";
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const int gate_failures = finite_gate_regression();
    if (argc == 2 && std::string_view(argv[1]) == "--finite-gate-only") {
        return gate_failures == 0 ? 0 : 1;
    }
    if (gate_failures != 0) { return 1; }
    const int width = argc > 1 ? std::atoi(argv[1]) : 1; // tokens per call: 1 = decode, 4 = MTP3
    if (argc > 2 && std::string_view(argv[2]) == "--two-stage") {
        // Prefill widths through the two-stage kernels, which stage an FP16 state through FP32.
#ifdef _WIN32
        _putenv_s("NINFER_GDN_TWO_STAGE", "1");
#else
        setenv("NINFER_GDN_TWO_STAGE", "1", 1);
#endif
    }
    try {
        std::mt19937 rng(0x5eed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<std::uint16_t> q(static_cast<std::size_t>(kDim) * kQk * kSteps);
        std::vector<std::uint16_t> k(q.size());
        std::vector<std::uint16_t> v(static_cast<std::size_t>(kDim) * kHv * kSteps);
        std::vector<float> g(static_cast<std::size_t>(kHv) * kSteps);
        std::vector<float> beta(g.size());
        for (auto& x : q) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : k) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : v) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (int t = 0; t < kSteps; ++t) {
            for (int h = 0; h < kHv; ++h) {
                const float alpha = 1.0F - std::pow(10.0F, -1.0F - 3.0F * h / (kHv - 1));
                g[static_cast<std::size_t>(t) * kHv + h] =
                    std::log(alpha) * (1.0F + 0.1F * normal(rng));
                beta[static_cast<std::size_t>(t) * kHv + h] = 1.0F / (1.0F + std::exp(-normal(rng)));
            }
        }
        ninfer::test::GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
        ninfer::test::GuardedDeviceBuffer dg(g.size() * 4), db(beta.size() * 4);
        dq.copy_from_host(q.data(), q.size() * 2);
        dk.copy_from_host(k.data(), k.size() * 2);
        dv.copy_from_host(v.data(), v.size() * 2);
        dg.copy_from_host(g.data(), g.size() * 4);
        db.copy_from_host(beta.data(), beta.size() * 4);
        const std::size_t state_elems = static_cast<std::size_t>(kDim) * kDim * kHv;
        ninfer::test::GuardedDeviceBuffer s32(state_elems * 4), s16(state_elems * 2);
        s32.fill(0);
        s16.fill(0);
        const std::size_t out_bytes = static_cast<std::size_t>(kDim) * kHv * width * 2;
        ninfer::test::GuardedDeviceBuffer o32(out_bytes), o16(out_bytes);
        ninfer::WorkspaceArena ws(std::max<std::size_t>(
            256, ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQk, kHv, true, width,
                                                                        width)));
        Tensor state32(s32.data(), DType::FP32, {kDim, kDim, kHv});
        Tensor state16(s16.data(), DType::FP16, {kDim, kDim, kHv});
        const float scale = 1.0F / std::sqrt(static_cast<float>(kDim));

        std::vector<std::uint16_t> h32(static_cast<std::size_t>(kDim) * kHv * width);
        std::vector<std::uint16_t> h16(h32.size());
        std::vector<double> err(kSteps / width);
        for (int call = 0; call < kSteps / width; ++call) {
            const std::size_t t0 = static_cast<std::size_t>(call) * width;
            const auto slice     = [&](void* base, std::size_t elem, std::size_t per_token,
                                   DType dtype, std::initializer_list<std::int32_t> shape) {
                return Tensor(static_cast<std::uint8_t*>(base) + t0 * per_token * elem, dtype,
                              shape);
            };
            const Tensor tq = slice(dq.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tk = slice(dk.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tv = slice(dv.data(), 2, kDim * kHv, DType::BF16, {kDim, kHv, width});
            const Tensor tg = slice(dg.data(), 4, kHv, DType::FP32, {kHv, width});
            const Tensor tb = slice(db.data(), 4, kHv, DType::FP32, {kHv, width});
            Tensor out32(o32.data(), DType::BF16, {kDim, kHv, width});
            Tensor out16(o16.data(), DType::BF16, {kDim, kHv, width});
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state32, out32,
                                         nullptr);
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state16, out16,
                                         nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "gated_delta_net");
            o32.copy_to_host(h32.data(), h32.size() * 2);
            o16.copy_to_host(h16.data(), h16.size() * 2);
            err[call] = relative_output_error(h32, h16, call);
        }
        const auto summary = summarize_errors(err);
        std::cout << "width " << width << ": relative output error, mean by quarter "
                  << summary.q1 << ' ' << summary.q2 << ' ' << summary.q3 << ' ' << summary.q4
                  << ", worst " << summary.worst << '\n';
        const bool ok = summary.passed();
        std::cout << (ok ? "OK" : "FAIL") << " FP16 GDN state tracks FP32 over " << kSteps
                  << " tokens\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP16 state test failed: " << error.what() << '\n';
        return 1;
    }
}

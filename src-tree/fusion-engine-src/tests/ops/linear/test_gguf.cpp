// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

struct TypeCase {
    QType qtype;
    const char* name;
    // Byte offsets of the binary16 multipliers inside one block, set to finite values.
    std::vector<int> half_fields;
};

std::vector<TypeCase> cases() {
    return {
        {QType::GGUF_Q8_0, "q8_0", {0}},
        {QType::GGUF_Q2_K, "q2_k", {80, 82}},
        {QType::GGUF_Q3_K, "q3_k", {108}},
        {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q5_K, "q5_k", {0, 2}},
        {QType::GGUF_Q6_K, "q6_k", {208}},
        {QType::GGUF_IQ2_XXS, "iq2_xxs", {0}},
        {QType::GGUF_IQ2_XS, "iq2_xs", {0}},
        {QType::GGUF_IQ2_S, "iq2_s", {0}},
        {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_IQ3_S, "iq3_s", {0}},
        {QType::GGUF_IQ1_S, "iq1_s", {0}},
        {QType::GGUF_IQ1_M, "iq1_m", {}},
        {QType::GGUF_IQ4_NL, "iq4_nl", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},
    };
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Random blocks with every binary16 multiplier finite; IQ1_M spreads its multiplier over the top
// nibbles of its four scale words.
std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.qtype == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
    }
    return bytes;
}

Weight gguf_weight(QType qtype, const void* data, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = qtype;
    w.layout  = QuantLayout::GgufBlocks;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = static_cast<std::uint32_t>(gguf_block_shape(qtype).elements);
    w.group      = gguf_block_shape(qtype).elements;
    return w;
}

std::vector<float> to_float(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = __bfloat162float(values[i]); }
    return out;
}

struct Error {
    double relative_rms = 0;
    double relative_max = 0;
};

Error compare(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = double(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
        worst = std::max(worst, std::abs(d));
        scale = std::max(scale, std::abs(want[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30)};
}

// Test-owned scalar decoding of represented GGUF weights. This does not call the production
// decoder, dequantize_rows, activation quantizer, or another Op to construct its reference.
double oracle_half_value(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    if (exponent == 31) { throw std::runtime_error("GGUF oracle: non-finite stored scale"); }
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -24)
                                           : std::ldexp(double(1024 + fraction), exponent - 25);
    return (bits & 0x8000) ? -magnitude : magnitude;
}

double q3_weight_value(const std::uint8_t* row, std::int32_t column) {
    constexpr int kBlockBytes = 110;
    const auto* block = row + (column / 256) * kBlockBytes;
    const int index = column % 256;
    const int half = index / 128, quarter = (index % 128) / 32, lane = index % 32;
    const int low = (block[32 + 32 * half + lane] >> (2 * quarter)) & 3;
    const bool positive = ((block[lane] >> (4 * half + quarter)) & 1) != 0;
    const int value = low - (positive ? 0 : 4);
    const int group = index / 16;
    const int low_scale = (block[96 + group % 8] >> (group >= 8 ? 4 : 0)) & 15;
    const int high_scale = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
    const int scale = (low_scale | (high_scale << 4)) - 32;
    const auto d = std::uint16_t(block[108]) | (std::uint16_t(block[109]) << 8);
    return oracle_half_value(static_cast<std::uint16_t>(d)) * double(scale) * double(value);
}

void check_q3_oracle_codec() {
    if (oracle_half_value(1) != std::ldexp(1.0, -24) || oracle_half_value(0x3c00) != 1.0 ||
        oracle_half_value(0xbc00) != -1.0 || oracle_half_value(0x7bff) != 65504.0) {
        throw std::runtime_error("Q3 oracle: independent binary16 self-check failed");
    }
    std::array<std::uint8_t, 110> block{};
    block[109] = 0x38; // exactly 0.5
    for (int group = 0; group < 16; ++group) {
        const int encoded = (group * 13) % 64;
        block[96 + group % 8] |= std::uint8_t((encoded & 15) << (group >= 8 ? 4 : 0));
        block[104 + group % 4] |= std::uint8_t((encoded >> 4) << (2 * (group / 4)));
    }
    for (int index = 0; index < 256; ++index) {
        const int value = (index * 5 + index / 32) % 8 - 4;
        const int half = index / 128, quarter = (index % 128) / 32, lane = index % 32;
        block[32 + 32 * half + lane] |= std::uint8_t((value & 3) << (2 * quarter));
        if (value >= 0) { block[lane] |= std::uint8_t(1 << (4 * half + quarter)); }
    }
    for (int index = 0; index < 256; ++index) {
        const double expected = 0.5 * ((index / 16 * 13) % 64 - 32) *
                                ((index * 5 + index / 32) % 8 - 4);
        if (q3_weight_value(block.data(), index) != expected) {
            throw std::runtime_error("Q3 oracle: independent packed-code self-check failed");
        }
    }
}

std::vector<double> q3_public_swiglu_oracle(
    const std::vector<std::uint8_t>& weights, const std::vector<std::uint16_t>& input,
    const std::vector<std::int32_t>& columns, bool gathered, std::int32_t rows,
    std::int32_t k, std::int32_t tokens, const std::vector<std::int32_t>& oracle_rows) {
    const std::size_t row_bytes = std::size_t(k / 256) * 110;
    std::vector<double> reference(std::size_t(tokens) * oracle_rows.size());
    for (int token = 0; token < tokens; ++token) {
        for (std::size_t sample = 0; sample < oracle_rows.size(); ++sample) {
            const int row = oracle_rows[sample];
            const auto* gate = weights.data() + std::size_t(row) * row_bytes;
            const auto* up = weights.data() + std::size_t(rows + row) * row_bytes;
            double gate_sum = 0.0, up_sum = 0.0;
            for (int column = 0; column < k; ++column) {
                const int source = gathered ? columns[column] : column;
                const double x = test::bf16_to_f32(input[std::size_t(token) * k + source]);
                gate_sum += q3_weight_value(gate, column) * x;
                up_sum += q3_weight_value(up, column) * x;
            }
            reference[std::size_t(token) * oracle_rows.size() + sample] =
                (gate_sum / (1.0 + std::exp(-gate_sum))) * up_sum;
        }
    }
    return reference; // No simulated q8 activation or BF16 intermediate/output rounding.
}

int run_q3_public_swiglu_case(std::int32_t rows, std::int32_t tokens,
                             bool gathered = false, bool replay = false,
                             bool offset_weights = false) {
    constexpr std::int32_t k = 5120;
    // Same public GGUF q8-activation SwiGLU criterion used by run_type below. It includes final
    // BF16 storage error; it is deliberately not the private represented-q8 probe's criterion.
    constexpr double kRelativeRms = 0.04, kRelativeMax = 0.16;
    std::mt19937 rng(0x513575u);
    const auto weights = random_blocks({QType::GGUF_Q3_K, "q3_k", {108}}, 2 * rows, k, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::uint16_t> represented(std::size_t(k) * tokens);
    for (auto& value : represented) { value = test::f32_to_bf16(normal(rng)); }
    std::vector<std::int32_t> columns(k);
    for (int i = 0; i < k; ++i) { columns[i] = i; }
    if (gathered) { std::shuffle(columns.begin(), columns.end(), rng); }

    std::vector<std::int32_t> oracle_rows;
    if (rows <= 128) {
        for (int row = 0; row < rows; ++row) { oracle_rows.push_back(row); }
    } else {
        oracle_rows = {0, 1, rows / 4, rows / 2 - 1, rows / 2, 3 * rows / 4, rows - 2, rows - 1};
    }
    // The real [34816,5120] parent runs at its full GPU shape. Its FP64 oracle samples complete
    // output rows: every input K and every token in each selected row. This is NOT an all-element
    // FP64 oracle at that shape. All GPU output elements still receive finite/canary checks.
    // N=37 covers vector tails. At T9 gguf_swiglu splits the parent into separate gate/up weights,
    // so N=128 makes EACH matrix product satisfy rows%128==0; parent extent alone is insufficient.
    // These small cases do not establish a real-N-only route.
    const std::string label = "Q3 public parent K5120 N=" + std::to_string(rows) +
        " T=" + std::to_string(tokens) + (gathered ? " gathered" : " contiguous") +
        (replay ? " graph" : " eager") + (offset_weights ? " weight_offset2" : " weight_aligned4");
    // Extend the prefix canary by two bytes, shifting only the weight payload from cudaMalloc's
    // aligned base. Every prefix/suffix byte remains guarded and the full original payload is
    // still compared below. The offset case must use the public route's 2-byte-aligned fallback.
    test::GuardedDeviceBuffer device_weights(weights.size(), offset_weights ? 258 : 256);
    if ((reinterpret_cast<std::uintptr_t>(device_weights.data()) & 3) !=
        (offset_weights ? 2U : 0U)) {
        throw std::runtime_error("Q3 oracle fixture: unexpected weight alignment");
    }
    test::GuardedDeviceBuffer device_input(represented.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_columns(columns.size() * sizeof(std::int32_t));
    test::GuardedDeviceBuffer device_output(std::size_t(rows) * tokens * sizeof(std::uint16_t));
    device_weights.copy_from_host(weights.data(), weights.size());
    device_input.copy_from_host(represented.data(), device_input.bytes());
    device_columns.copy_from_host(columns.data(), device_columns.bytes());
    Weight parent = gguf_weight(QType::GGUF_Q3_K, device_weights.data(), 2 * rows, k);
    if (gathered) { parent.input_columns = static_cast<const std::int32_t*>(device_columns.data()); }
    Tensor x(device_input.data(), DType::BF16, {k, tokens});
    Tensor out(device_output.data(), DType::BF16, {rows, tokens});
    const auto capacity = ops::linear_swiglu_workspace_capacity_bytes(
        QType::GGUF_Q3_K, 2 * rows, k, ops::LinearPolicy::A16Only, tokens, tokens);
    test::GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
    DeviceContext context;
    const auto launch = [&] {
        ops::linear_swiglu(x, parent, out, ops::LinearPolicy::A16Only, workspace, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        launch(); // Initialize any lazy launch metadata before capture.
        test::cuda_synchronize(context.stream);
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase != 0) {
            for (auto& bits : represented) { bits ^= 0x8000; }
            device_input.copy_from_host(represented.data(), device_input.bytes());
        }
        device_output.fill(0xff); // Poison with BF16 NaNs so omitted output stores cannot pass.
        if (replay) { graph.launch(context.stream); }
        else { launch(); }
        test::cuda_synchronize(context.stream);
        const auto actual_bits = test::from_device<std::uint16_t>(device_output.data(), std::size_t(rows) * tokens);
        const bool finite = std::all_of(actual_bits.begin(), actual_bits.end(), [](std::uint16_t bits) {
            return std::isfinite(test::bf16_to_f32(bits));
        });
        const auto reference = q3_public_swiglu_oracle(weights, represented, columns, gathered,
                                                       rows, k, tokens, oracle_rows);
        std::vector<float> actual;
        actual.reserve(reference.size());
        for (int token = 0; token < tokens; ++token) {
            for (const int row : oracle_rows) {
                actual.push_back(test::bf16_to_f32(actual_bits[std::size_t(token) * rows + row]));
            }
        }
        const Error error = compare(actual, reference);
        const bool ok = finite && std::all_of(reference.begin(), reference.end(), [](double v) {
            return std::isfinite(v);
        }) && error.relative_rms < kRelativeRms && error.relative_max < kRelativeMax;
        std::cout << (ok ? "OK   " : "FAIL ") << label << " replay_phase=" << phase
                  << " oracle_complete_rows=" << oracle_rows.size() << "/" << rows
                  << " rms=" << error.relative_rms << " max=" << error.relative_max << '\n';
        failures += ok ? 0 : 1;
        failures += device_weights.verify_guards(label + " weight");
        failures += device_input.verify_guards(label + " input");
        failures += device_columns.verify_guards(label + " gather map");
        failures += device_output.verify_guards(label + " output");
        failures += scratch.verify_guards(label + " workspace");
        if (test::from_device<std::uint8_t>(device_weights.data(), weights.size()) != weights ||
            test::from_device<std::uint16_t>(device_input.data(), represented.size()) != represented ||
            test::from_device<std::int32_t>(device_columns.data(), columns.size()) != columns) {
            std::cerr << label << ": weight, input or gather map was modified\n";
            ++failures;
        }
        if (workspace.peak_used() > capacity || workspace.used() != 0) {
            std::cerr << label << ": workspace query/scope mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int run_q3_public_oracle_cases() {
    check_q3_oracle_codec();
    int failures = 0;
    for (const int tokens : {1, 4, 5, 6, 8}) {
        failures += run_q3_public_swiglu_case(37, tokens);
    }
    failures += run_q3_public_swiglu_case(128, 8);
    failures += run_q3_public_swiglu_case(128, 9);
    failures += run_q3_public_swiglu_case(37, 5, true, true);
    failures += run_q3_public_swiglu_case(37, 5, false, false, true);
    failures += run_q3_public_swiglu_case(17408, 5);
    failures += run_q3_public_swiglu_case(17408, 5, false, true);
    return failures;
}

double q6_weight_value(const std::uint8_t* row, std::int32_t column) {
    const auto* block = row + (column / 256) * 210;
    const int index = column % 256;
    const int half = index / 128, quarter = (index % 128) / 32, lane = index % 32;
    const int low = (block[64 * half + 32 * (quarter & 1) + lane] >>
                     (4 * (quarter / 2))) & 15;
    const int high = (block[128 + 32 * half + lane] >> (2 * quarter)) & 3;
    const int value = (low | (high << 4)) - 32;
    const int stored_scale = block[192 + index / 16];
    const int scale = stored_scale < 128 ? stored_scale : stored_scale - 256;
    const auto d = std::uint16_t(block[208]) | (std::uint16_t(block[209]) << 8);
    return oracle_half_value(static_cast<std::uint16_t>(d)) * double(scale) * double(value);
}

void check_q6_oracle_codec() {
    std::array<std::uint8_t, 210> block{};
    block[209] = 0x38; // exactly 0.5
    for (int group = 0; group < 16; ++group) {
        block[192 + group] = std::uint8_t((group * 37 + 128) & 255);
    }
    for (int index = 0; index < 256; ++index) {
        const int code = (index * 7 + index / 16) % 64;
        const int half = index / 128, quarter = (index % 128) / 32, lane = index % 32;
        block[64 * half + 32 * (quarter & 1) + lane] |= std::uint8_t((code & 15) << (4 * (quarter / 2)));
        block[128 + 32 * half + lane] |= std::uint8_t((code >> 4) << (2 * quarter));
    }
    for (int index = 0; index < 256; ++index) {
        const int stored_scale = (index / 16 * 37 + 128) & 255;
        const int scale = stored_scale < 128 ? stored_scale : stored_scale - 256;
        const double expected = 0.5 * scale * ((index * 7 + index / 16) % 64 - 32);
        if (q6_weight_value(block.data(), index) != expected) {
            throw std::runtime_error("Q6 oracle: independent packed-code self-check failed");
        }
    }
}

// Scalar element decoding, independent of the vector decoder's packed word extraction.
double q5_weight_value(const std::uint8_t* row, std::int32_t column) {
    const auto* b = row + (column / 256) * 176;
    const int index = column % 256, group = index / 32, lane = index % 32;
    const int code = ((b[48 + 32 * (group / 2) + lane] >> (4 * (group & 1))) & 15) |
                     (((b[16 + lane] >> group) & 1) << 4);
    int scale, minimum;
    if (group < 4) {
        scale = b[4 + group] & 63;
        minimum = b[8 + group] & 63;
    } else {
        scale = (b[8 + group] & 15) + 16 * (b[group] / 64);
        minimum = (b[8 + group] / 16) + 16 * (b[4 + group] / 64);
    }
    const auto d = std::uint16_t(b[0]) | (std::uint16_t(b[1]) << 8);
    const auto dm = std::uint16_t(b[2]) | (std::uint16_t(b[3]) << 8);
    return oracle_half_value(static_cast<std::uint16_t>(d)) * scale * code -
           oracle_half_value(static_cast<std::uint16_t>(dm)) * minimum;
}

void check_q5_oracle_codec() {
    std::array<std::uint8_t, 176> block{};
    block[1] = 0x38; // d = 0.5, dmin = 0.25; both exactly represented.
    block[3] = 0x34;
    for (int group = 0; group < 8; ++group) {
        const int scale = (11 * group + 7) % 64, minimum = (17 * group + 3) % 64;
        if (group < 4) {
            block[4 + group] |= std::uint8_t(scale);
            block[8 + group] |= std::uint8_t(minimum);
        } else {
            block[8 + group] |= std::uint8_t((scale % 16) + 16 * (minimum % 16));
            block[group] |= std::uint8_t((scale / 16) * 64);
            block[4 + group] |= std::uint8_t((minimum / 16) * 64);
        }
    }
    for (int index = 0; index < 256; ++index) {
        const int group = index / 32, lane = index % 32, code = (13 * index + group) % 32;
        block[48 + 32 * (group / 2) + lane] |= std::uint8_t((code % 16) << (4 * (group & 1)));
        block[16 + lane] |= std::uint8_t((code / 16) << group);
    }
    for (int index = 0; index < 256; ++index) {
        const int group = index / 32;
        const double expected = 0.5 * ((11 * group + 7) % 64) * ((13 * index + group) % 32) -
                                0.25 * ((17 * group + 3) % 64);
        if (q5_weight_value(block.data(), index) != expected) {
            throw std::runtime_error("Q5 oracle: independent packed-code/minimum self-check failed");
        }
    }
}

// The public Linear oracle retains the original BF16 activation values and independent Q5/Q6
// weight decoding. q8_1 codes and their FP16 scales are private implementation error, covered by
// the same Linear tolerance as run_type; no q8 quantizer or simulated cast is used as the oracle.
enum class KTestOp { Linear, Add, SwigluPair };

int run_k_linear_oracle_case(QType qtype, std::int32_t rows, std::int32_t k, std::int32_t tokens,
                              bool replay = false, bool offset_weights = false,
                              int padded_blocks = 0, bool gathered = false,
                              KTestOp op = KTestOp::Linear) {
    if (qtype != QType::GGUF_Q5_K && qtype != QType::GGUF_Q6_K) {
        throw std::runtime_error("K oracle fixture: unsupported type");
    }
    const bool q5 = qtype == QType::GGUF_Q5_K;
    const TypeCase encoding = q5 ? TypeCase{qtype, "Q5", {0, 2}} : TypeCase{qtype, "Q6", {208}};
    const int block_bytes = q5 ? 176 : 210;
    const auto weight_value = q5 ? q5_weight_value : q6_weight_value;
    // Q5's existing code windows require 16-byte alignment; Q6 permits two-byte alignment.
    const int weight_offset = offset_weights ? (q5 ? 16 : 2) : 0;
    const bool paired = op == KTestOp::SwigluPair;
    const double kRelativeRms = paired ? 0.04 : 0.02, kRelativeMax = paired ? 0.16 : 0.08;
    const int weight_rows = paired ? 2 * rows : rows;
    std::mt19937 rng(0x615575u);
    const auto dense_weights = random_blocks(encoding, weight_rows, k, rng);
    const std::size_t dense_stride = std::size_t(k / 256) * block_bytes;
    const std::size_t row_stride = dense_stride + std::size_t(padded_blocks) * block_bytes;
    std::vector<std::uint8_t> weights(std::size_t(weight_rows) * row_stride, 0xa5);
    for (int row = 0; row < weight_rows; ++row) {
        std::memcpy(weights.data() + std::size_t(row) * row_stride,
                    dense_weights.data() + std::size_t(row) * dense_stride, dense_stride);
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::uint16_t> represented(std::size_t(k) * tokens);
    for (auto& value : represented) { value = test::f32_to_bf16(normal(rng)); }
    // Include a complete zero activation group, exercising the private quantizer's zero scale.
    for (int token = 0; token < tokens; ++token) {
        std::fill_n(represented.begin() + std::size_t(token) * k, 32, std::uint16_t{0});
    }
    std::vector<std::uint16_t> residual;
    if (op == KTestOp::Add) {
        residual.resize(std::size_t(rows) * tokens);
        for (auto& value : residual) { value = test::f32_to_bf16(32.0f * normal(rng)); }
    }
    std::vector<std::int32_t> columns(k);
    for (int i = 0; i < k; ++i) { columns[i] = i; }
    if (gathered) { std::shuffle(columns.begin(), columns.end(), rng); }
    std::vector<std::int32_t> oracle_rows;
    if (rows <= 128) {
        for (int row = 0; row < rows; ++row) { oracle_rows.push_back(row); }
    } else {
        oracle_rows = {0, 1, rows / 4, rows / 2 - 1, rows / 2, 3 * rows / 4, rows - 2, rows - 1};
    }
    // Real N5120 cases sample eight complete output rows (all K and T), not all-element FP64.
    // Every output still receives finite/canary checks. Small cases check every output row.
    const std::string label = std::string(encoding.name) + (padded_blocks ? " bridge padded" :
        paired ? " public swiglu pair" : op == KTestOp::Add ? " public linear_add" : " public linear") +
        " N=" + std::to_string(rows) + " K=" + std::to_string(k) + " T=" + std::to_string(tokens) +
        " stride=" + std::to_string(row_stride) + (replay ? " graph" : " eager") +
        (" weight_offset=" + std::to_string(weight_offset)) + (gathered ? " gathered" : "");
    test::GuardedDeviceBuffer device_weights(weights.size(), 256 + weight_offset);
    if ((reinterpret_cast<std::uintptr_t>(device_weights.data()) & 255) != unsigned(weight_offset)) {
        throw std::runtime_error("K oracle fixture: unexpected weight alignment");
    }
    test::GuardedDeviceBuffer device_input(represented.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_columns(columns.size() * sizeof(std::int32_t));
    test::GuardedDeviceBuffer device_output(std::size_t(rows) * tokens * sizeof(std::uint16_t));
    device_weights.copy_from_host(weights.data(), weights.size());
    device_input.copy_from_host(represented.data(), device_input.bytes());
    device_columns.copy_from_host(columns.data(), device_columns.bytes());
    Weight weight = gguf_weight(qtype, device_weights.data(), rows, k);
    if (gathered) { weight.input_columns = static_cast<const std::int32_t*>(device_columns.data()); }
    Weight up = weight;
    if (paired) {
        up.qdata = up.payload = static_cast<const std::uint8_t*>(weight.qdata) + std::size_t(rows) * row_stride;
    }
    Tensor x(device_input.data(), DType::BF16, {k, tokens});
    Tensor out(device_output.data(), DType::BF16, {rows, tokens});
    const auto capacity = paired ? ops::linear_swiglu_pair_workspace_capacity_bytes(
        qtype, qtype, rows, k, ops::LinearPolicy::A16Only, tokens, tokens) :
        op == KTestOp::Add ? ops::linear_add_workspace_capacity_bytes(
        qtype, rows, k, ops::LinearPolicy::A16Only, tokens, tokens) :
        ops::linear_workspace_capacity_bytes(
        qtype, rows, k, ops::LinearPolicy::A16Only, tokens, tokens);
    test::GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
    DeviceContext context;
    std::optional<test::GuardedDeviceBuffer> device_dense;
    if (padded_blocks) {
        // Public Weight stores dense GGUF rows. The bridge owns the supported padded stride,
        // so test that boundary explicitly, including exact equivalence to the public dense case.
        if (op != KTestOp::Linear || tokens > ops::gguf::kMaxVectorColumns ||
            ops::gguf::vector_activation_bytes(k, tokens) > capacity) {
            throw std::runtime_error("Q6 padded fixture exceeds vector workspace");
        }
        device_dense.emplace(dense_weights.size());
        device_dense->copy_from_host(dense_weights.data(), dense_weights.size());
    }
    const auto launch = [&] {
        if (padded_blocks) {
            ops::gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data),
                k, tokens, weight.input_columns, scratch.data(), context.stream);
            const ops::gguf::VectorOutput output{static_cast<__nv_bfloat16*>(out.data), nullptr,
                                                 rows, ops::gguf::Epilogue::Store};
            ops::gguf::vector_product(ops::detail::gguf_type(qtype), weight.qdata, row_stride,
                                      rows, k, scratch.data(), tokens, output, context.stream);
        } else if (paired) {
            ops::linear_swiglu(x, weight, up, out, ops::LinearPolicy::A16Only, workspace, context.stream);
        } else if (op == KTestOp::Add) {
            ops::linear_add(x, weight, out, ops::LinearPolicy::A16Only, workspace, context.stream);
        } else {
            ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, context.stream);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        if (!residual.empty()) { device_output.copy_from_host(residual.data(), device_output.bytes()); }
        launch();
        test::cuda_synchronize(context.stream);
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase != 0) {
            for (auto& bits : represented) { bits ^= 0x8000; }
            device_input.copy_from_host(represented.data(), device_input.bytes());
        }
        std::vector<std::uint16_t> dense_output;
        if (device_dense) {
            Weight dense = weight;
            dense.qdata = dense.payload = device_dense->data();
            ops::linear(x, dense, out, ops::LinearPolicy::A16Only, workspace, context.stream);
            test::cuda_synchronize(context.stream);
            dense_output = test::from_device<std::uint16_t>(out.data, std::size_t(rows) * tokens);
        }
        if (residual.empty()) { device_output.fill(0xff); }
        else { device_output.copy_from_host(residual.data(), device_output.bytes()); }
        if (replay) { graph.launch(context.stream); } else { launch(); }
        test::cuda_synchronize(context.stream);
        const auto actual_bits = test::from_device<std::uint16_t>(out.data, std::size_t(rows) * tokens);
        std::vector<float> actual;
        std::vector<double> reference;
        for (int token = 0; token < tokens; ++token) {
            for (const int row : oracle_rows) {
                double sum = 0.0, upper = 0.0;
                for (int column = 0; column < k; ++column) {
                    const int source = gathered ? columns[column] : column;
                    const double value = test::bf16_to_f32(represented[std::size_t(token) * k + source]);
                    sum += weight_value(weights.data() + std::size_t(row) * row_stride, column) *
                           value;
                    if (paired) {
                        upper += weight_value(weights.data() + std::size_t(rows + row) * row_stride, column) * value;
                    }
                }
                if (paired) { sum = (sum / (1.0 + std::exp(-sum))) * upper; }
                if (!residual.empty()) { sum += test::bf16_to_f32(residual[std::size_t(token) * rows + row]); }
                reference.push_back(sum);
                actual.push_back(test::bf16_to_f32(actual_bits[std::size_t(token) * rows + row]));
            }
        }
        const Error error = compare(actual, reference);
        const bool finite = std::all_of(actual_bits.begin(), actual_bits.end(), [](std::uint16_t bits) {
            return std::isfinite(test::bf16_to_f32(bits));
        }) && std::all_of(reference.begin(), reference.end(), [](double value) { return std::isfinite(value); });
        const bool dense_exact = !device_dense || actual_bits == dense_output;
        const bool ok = finite && dense_exact && error.relative_rms < kRelativeRms &&
                        error.relative_max < kRelativeMax;
        std::cout << (ok ? "OK   " : "FAIL ") << label << " replay_phase=" << phase
                  << " oracle_complete_rows=" << oracle_rows.size() << "/" << rows
                  << " dense_exact=" << dense_exact << " rms=" << error.relative_rms
                  << " max=" << error.relative_max << '\n';
        failures += ok ? 0 : 1;
        failures += device_weights.verify_guards(label + " weight");
        failures += device_input.verify_guards(label + " input");
        failures += device_columns.verify_guards(label + " gather map");
        failures += device_output.verify_guards(label + " output");
        failures += scratch.verify_guards(label + " workspace");
        if (device_dense) { failures += device_dense->verify_guards(label + " dense weight"); }
        if (test::from_device<std::uint8_t>(device_weights.data(), weights.size()) != weights ||
            test::from_device<std::uint16_t>(device_input.data(), represented.size()) != represented ||
            test::from_device<std::int32_t>(device_columns.data(), columns.size()) != columns ||
            (device_dense && test::from_device<std::uint8_t>(device_dense->data(), dense_weights.size()) != dense_weights)) {
            std::cerr << label << ": weight, input or gather map was modified\n";
            ++failures;
        }
        if (workspace.peak_used() > capacity || workspace.used() != 0) {
            std::cerr << label << ": workspace query/scope mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int run_q6_public_oracle_cases() {
    check_q6_oracle_codec();
    int failures = 0;
    for (const int tokens : {4, 5, 6}) { failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 37, 6144, tokens); }
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 37, 17408, 5);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 128, 6144, 8);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 128, 6144, 9);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 37, 6144, 5, true, false, 0, true);
    for (const int tokens : {4, 6}) { failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, tokens); }
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, 5, true, false, 0, true);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, 5, true, false, 0, false, KTestOp::Add);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, 5, true, false, 0, false, KTestOp::SwigluPair);
    for (const int k : {6144, 17408}) {
        failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, k, 5);
        failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, k, 5, true);
        failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, k, 5, false, true);
    }
    // Padding by one/two blocks covers both 2-byte and 4-byte row strides at the real N/K gate.
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, 5, false, false, 1);
    failures += run_k_linear_oracle_case(QType::GGUF_Q6_K, 5120, 6144, 5, false, false, 2);
    return failures;
}

int run_q5_public_oracle_cases() {
    check_q5_oracle_codec();
    int failures = 0;
    const auto type = QType::GGUF_Q5_K;
    // Full small matrices cover all scale/min groups and tail rows; large cases sample all K
    // of eight rows while checking every output for finite values and all buffers for canaries.
    failures += run_k_linear_oracle_case(type, 37, 6144, 5);
    failures += run_k_linear_oracle_case(type, 37, 17408, 5);
    for (const int tokens : {4, 6}) { failures += run_k_linear_oracle_case(type, 5120, 6144, tokens); }
    for (const int tokens : {8, 9}) { failures += run_k_linear_oracle_case(type, 128, 6144, tokens); }
    for (const int k : {6144, 17408}) {
        failures += run_k_linear_oracle_case(type, 5120, k, 5);
        failures += run_k_linear_oracle_case(type, 5120, k, 5, true, false, 0, true);
    }
    failures += run_k_linear_oracle_case(type, 5120, 6144, 5, true, false, 0, false, KTestOp::Add);
    failures += run_k_linear_oracle_case(type, 5120, 6144, 5, true, false, 0, false, KTestOp::SwigluPair);
    failures += run_k_linear_oracle_case(type, 5120, 6144, 5, false, true);
    // A padded aligned row must keep the generic route and match the dense candidate exactly.
    failures += run_k_linear_oracle_case(type, 5120, 6144, 5, false, false, 1);
    failures += run_k_linear_oracle_case(type, 5120, 5120, 5); // unqualified K fallback
    return failures;
}

int run_type(const TypeCase& c, std::uint32_t seed) {
    constexpr std::int32_t kRows = 256;
    constexpr std::int32_t kK    = 1536;
    const std::int32_t widths[]  = {1, 2, 3, 5, 8, 9, 17, 40, 70, 130};
    const std::int32_t max_t     = 130;
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, kK, rng);
    DeviceBuffer blocks(host_blocks.size());
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block        = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(kK / block.elements) * block.bytes;
    const Weight gate = gguf_weight(c.qtype, blocks.p, kRows, kK);
    Weight up         = gate;
    up.qdata          = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;
    const Weight both = gguf_weight(c.qtype, blocks.p, 2 * kRows, kK);

    // Exact weights.
    DeviceBuffer exact_device(std::size_t(2) * kRows * kK * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, kK,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), kK,
                               nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * kK);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    for (const float v : exact) {
        if (!std::isfinite(v)) {
            std::cerr << c.name << ": non-finite dequantized weight\n";
            return 1;
        }
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(kK) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    const auto x_float = to_float(x_host);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> columns(kK);
    for (std::int32_t i = 0; i < kK; ++i) { columns[i] = i; }
    std::shuffle(columns.begin(), columns.end(), rng);
    DeviceBuffer columns_device(columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(columns_device.p);

    const auto product = [&](std::int32_t row0, std::int32_t t, bool permuted) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                double acc = 0;
                const float* w = exact.data() + std::size_t(row0 + r) * kK;
                const float* x = x_float.data() + std::size_t(col) * kK;
                for (std::int32_t i = 0; i < kK; ++i) {
                    acc += double(w[i]) * x[permuted ? columns[i] : i];
                }
                out[std::size_t(col) * kRows + r] = acc;
            }
        }
        return out;
    };

    const std::size_t capacity = std::max(
        {ops::linear_workspace_capacity_bytes(c.qtype, kRows, kK, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, kK,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, kK,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    // q8_1 activations cost one product about 1 % RMS against the exact weights; a SwiGLU output
    // carries two such products, so it gets twice the budget.
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want) {
        const Error e     = compare(got, want);
        const double gain = label.starts_with("swiglu") ? 2.0 : 1.0;
        const bool ok     = e.relative_rms < 2e-2 * gain && e.relative_max < 8e-2 * gain;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " " << label << " rms=" << e.relative_rms
                      << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {kK, t});
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);

        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        const auto plain = product(0, t, false);
        failures += run_check("linear T=" + std::to_string(t), to_float(host), plain);

        ops::linear(x, gathered, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "gathered linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("gathered T=" + std::to_string(t), to_float(host),
                              product(0, t, true));

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add T=" + std::to_string(t), to_float(host), sum);

        const auto upper = product(kRows, t, false);
        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu T=" + std::to_string(t), to_float(host), swiglu);

        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair T=" + std::to_string(t), to_float(host), swiglu);
    }

    // The token-table gather is the exact weights rounded to BF16.
    const std::vector<std::int32_t> ids{3, 0, 255, 17, 3};
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer rows_device(ids.size() * kK * sizeof(__nv_bfloat16));
    Tensor ids_tensor(ids_device.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor rows(rows_device.p, DType::BF16, {kK, static_cast<std::int32_t>(ids.size())});
    ops::embedding(ids_tensor, gate, rows, nullptr);
    check(cudaDeviceSynchronize(), "embedding");
    std::vector<__nv_bfloat16> gathered_rows(ids.size() * kK);
    rows_device.copy_to_host(gathered_rows.data(), gathered_rows.size() * sizeof(__nv_bfloat16));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const float want = __bfloat162float(__float2bfloat16(exact[std::size_t(ids[i]) * kK + j]));
            if (__bfloat162float(gathered_rows[i * kK + j]) != want) {
                std::cerr << "FAIL " << c.name << " embedding row " << ids[i] << " column " << j
                          << '\n';
                ++failures;
                i = ids.size();
                break;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures      = run_q3_public_oracle_cases() + run_q6_public_oracle_cases() + run_q5_public_oracle_cases();
        std::uint32_t seed = 9001;
        for (const auto& c : cases()) {
            const int f = run_type(c, seed++);
            std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << '\n';
            failures += f;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF Linear: " << error.what() << '\n';
        return 1;
    }
}

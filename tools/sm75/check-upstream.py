"""Read upstream state without merging, checking out, or changing runtime files."""
import argparse
import datetime
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def get(url):
    request = urllib.request.Request(url, headers={
        'User-Agent': 'ninfer-sm75-upstream-check',
        'Accept': 'application/vnd.github+json', 'Cache-Control': 'no-cache'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    baseline = json.loads((ROOT/'sm75/baseline.json').read_text(encoding='utf-8'))
    api = 'https://api.github.com/repos/' + baseline['upstream_repository']
    head = get(api+'/commits/main')
    releases = get(api+'/releases?per_page=1')
    comparison = get(api+'/compare/'+baseline['upstream_commit']+'...'+head['sha'])
    result = {'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'baseline':baseline['upstream_commit'], 'upstream_head':head['sha'],
              'new_commits':comparison['ahead_by'], 'status':comparison['status'],
              'needs_review':head['sha']!=baseline['upstream_commit'],
              'latest_release':releases[0]['tag_name'] if releases else None,
              'compare_url':comparison['html_url'],
              'commits':[{'sha':c['sha'],'subject':c['commit']['message'].splitlines()[0],
                          'url':c['html_url']} for c in comparison['commits']]}
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

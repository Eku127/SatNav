"""Merge the separately checked-out website with the built multilingual Wiki."""
from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--website', type=Path, required=True)
    parser.add_argument('--docs', type=Path, default=Path('docs/_build/html'))
    args = parser.parse_args()
    if not (args.website / 'index.html').is_file():
        raise FileNotFoundError('Website index.html is missing')
    if not (args.docs / 'wiki/en-US/index.html').is_file():
        raise FileNotFoundError('Build the Wiki before assembling Pages')
    reserved = {'wiki', 'en-US', 'zh-CN'}
    for item in args.website.iterdir():
        if item.name in reserved:
            raise ValueError(f'Website uses reserved Wiki path: {item.name}')
    shutil.copytree(args.website, args.docs, dirs_exist_ok=True)
    (args.docs / '.nojekyll').touch()
    print(f'Website and Wiki ready: {args.docs}')


if __name__ == '__main__':
    main()

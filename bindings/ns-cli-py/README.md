# nextstat-cli

Command-line interface for [NextStat](https://pypi.org/project/nextstat/) — a high-performance statistical inference engine.

## Installation

```bash
pip install nextstat-cli
```

Or install together with the Python library (recommended):

```bash
pip install nextstat          # automatically pulls nextstat-cli
```

## Usage

```bash
nextstat version
nextstat fit --input workspace.json
nextstat hypotest --input workspace.json --mu 1.0
nextstat upper-limit --input workspace.json --expected
```

For full documentation, see <https://nextstat.io/docs/cli>.

## License

AGPL-3.0-or-later OR commercial — see [LICENSE](https://github.com/NextStat/nextstat.io/blob/main/LICENSE).

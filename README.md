# MariUdmurt

Adapt [Glot500](https://huggingface.co/cis-lmu/glot500-base) for **Meadow/Eastern
Mari and standard Udmurt UD parsing**, and future **Nganasan morphological parsing**.
Enets, Nenets, and Selkup provide related Samoyedic data for the Nganasan objective;
their transfer benefit will be evaluated. The initial experiment samples Mari,
Udmurt, and Nganasan at 30% each, with the remaining 10% shared by the other languages.

| Local corpus | Source |
| --- | --- |
| Mari (`mhr`), Udmurt (`udm`) | [Mari Wikipedia](https://mhr.wikipedia.org/), [Udmurt Wikipedia](https://udm.wikipedia.org/) dump extracts |
| Enets, Nenets, Nganasan | [INEL, Universität Hamburg repository](https://www.fdr.uni-hamburg.de/communities/inel/) |
| Selkup | [INEL Selkup corpus portal](https://inel.corpora.uni-hamburg.de/portal/corpora/selkup/) |

Python scripts prepare source-preserving splits, train/evaluate the backbone, and
record reproducible experiments. See [backbone setup and evaluation](docs/backbone.md).
The [joint UD parser workflow](docs/parser.md) trains on Finnish, Hungarian,
Erzya, Moksha, Estonian, Komi Permyak, Komi Zyrian, and Tundra Nenets from
[UD 2.18](https://universaldependencies.org/download.html) for zero-shot
Mari/Udmurt transfer. See the [PowerShell training and evaluation
instructions](uralic_parser/instructions.md). Human-testing notebooks are
generated locally and ignored by Git. Future work includes controlled
transliteration/transcription experiments.

Parser code adapted from [BaseUDParser](https://github.com/fortvivlan/BaseUDParser)
is GPL-3.0; see [LICENSE](LICENSE). The bundled UD scorer retains MPL-2.0 terms.

Conducted under **RSF Grant No. 25-18-00222**
([project website](https://fortvivlan.github.io/controlandraise/index_en.html)).

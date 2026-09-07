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
record reproducible experiments. See [setup, commands, provenance, and evaluation](docs/backbone.md).
Human-testing notebooks are generated locally and ignored by Git. Future work
includes controlled transliteration/transcription experiments and downstream
training using [BaseUDParser](https://github.com/fortvivlan/BaseUDParser).

Conducted under **RSF Grant No. 25-18-00222**
([project website](https://fortvivlan.github.io/controlandraise/index_en.html)).

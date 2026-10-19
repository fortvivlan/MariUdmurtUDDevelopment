# Meadow Mari UD Annotation Guidelines — Pilot Draft

> **Status:** working reference for manual annotation, not a frozen specification. Reorganized from the project notes (19 October 2026). The decisions below have **not** been independently rechecked against the grammar or current UD release. Keep disputed cases visible until adjudication.
>
> **Labels:** **ADOPT** = intended pilot convention; **CONTEXT** = depends on use; **PROVISIONAL** = tentative; **EXPERT QUESTION** = explicitly marked for consultation; **REVIEW** = another open consistency or corpus question.
>
> **Notation:** `_` = feature absent. Sample feature strings show only the features relevant to the issue; other morphologically justified features may also apply. POS labels are uppercase (`NOUN`, `AUX`, etc.).

## Contents

1. [Cross-cutting principles](#1-cross-cutting-principles)
2. [Nouns and temporal nominals](#2-nouns-and-temporal-nominals)
3. [Adjectives, degree and adjectival quantifiers](#3-adjectives-degree-and-adjectival-quantifiers)
4. [Numerals](#4-numerals)
5. [Pronouns, determiners and pro-adverbs](#5-pronouns-determiners-and-pro-adverbs)
6. [Adpositions](#6-adpositions)
7. [Finite verbs, tense, negation and existential predicates](#7-finite-verbs-tense-negation-and-existential-predicates)
8. [Non-finite verb forms](#8-non-finite-verb-forms)
9. [Auxiliaries and multi-verb constructions](#9-auxiliaries-and-multi-verb-constructions)
10. [Particles](#10-particles)
11. [Dependency relations](#11-dependency-relations)
12. [Questions and checks](#12-questions-and-checks)
13. [Feature inventory and references](#13-feature-inventory-and-references)

---

## Annotation quick lookup

| Phenomenon | Working decision |
|---|---|
| Temporal case-marked nouns | `NOUN` + actual `Case` + `obl:tmod` |
| Color adjectives with short/long opposition | both `Variant=Short` and `Variant=Long` |
| Analytic superlative `эн + ADJ` | `эн ADV Degree=Sup` → `advmod` |
| Group numerals ‘N of us/you/them’ | `NUM NumType=Assoc` |
| Polite `те` | `Polite=Form` only with unambiguous formal address |
| Genitive personal pronouns | `Case=Gen`; no `Poss=Yes` |
| `ни-` negative pronouns/adverbs | `PronType=Neg`, not `Polarity=Neg` |
| Negative auxiliary + connegative | `AUX Polarity=Neg` + `VERB Connegative=Yes` |
| Past II | `Tense=Past` + provisional `Evident=Nfh` + `VerbForm=Fin` |
| Converb subclasses | `ConvType=Gen/Ant/Post/Sim` + `VerbForm=Conv` |
| Grammaticalized aspectual verb | `AUX`, `aux:aspect`, normal inflection |
| `уке` ‘there is not’ | provisional `ADJ Polarity=Neg`; **expert question E1** |

---

## 1. Cross-cutting principles

- **ADOPT — Morphology vs. function:** preserve a token's morphological category even if the phrase functions adverbially (e.g. case-marked temporal `NOUN` with `obl:tmod`).
- **ADOPT — Do not infer word-level tense from constructional time reference.** In an analytic future, the auxiliary can bear morphological `Tense=Pres` while the construction refers to the future.
- **ADOPT — Syntactic auxiliaries:** tag a sufficiently grammaticalized functional verb `AUX`; use its overt person/number/mood/tense morphology. A homonymous lexical use remains `VERB`.
- **ADOPT — Possessive marking:** distinguish possessors (`PRON Case=Gen`), possessor indexing on the possessed word (`Person[psor]`, `Number[psor]`), and syntactic possession (`nmod:poss`). Do **not** add `Poss=Yes` merely for a genitive personal pronoun.
- **ADOPT — Pro-form series:** distinguish `PronType=Int`, `Neg`, `Ind`, `Tot`, `Dem`, `Prs`, `Rcp` by function and paradigm; `PronType` is not restricted to `PRON` UPOS.
- **ADOPT — Rich local morphology is permitted:** pilot-specific `Variant`, `InfForm`, `PartForm`, `ConvType`, `NumType=Assoc`, etc., may be added even without a matching transfer-language feature. The FST/manual correction stage can supply them.
- **PROVISIONAL — Prioritize reproducible criteria:** where lexicalization or grammaticalization is unclear, record the ambiguity rather than inferring UPOS solely from a dictionary's traditional label.

## 2. Nouns and temporal nominals

### 2.1. Nominal features

| Feature | Working values and morphology | Status |
|---|---|---|
| `Number` | `Sing`, `Plur`; plural markers `-влак`, `-шамыч`, `-ла`, `-мыт` | **ADOPT** |
| `Number[psor]` | `Sing`, `Plur` | **ADOPT** |
| `Person[psor]` | `1`, `2`, `3` | **ADOPT** |
| `Case` | `Nom`; `Acc` (`-м`); `Gen` (`-н`); `Dat` (`-лан`); `Ine` (`-ште`); `Ill` (`-ш(ке)`); `Lat` (`-еш`); `Cmp` (`-ла`); `Com` (`-ге`) | **ADOPT**, but verify inventory/suffix mapping before release |

Case meanings in the notes: `Ine` = location inside something; `Ill` = direction into; `Lat` = movement toward/to/into/onto; `Cmp` = comparative case; `Com` = comitative case.

### 2.2. Temporal expressions

**ADOPT:** a regular case-marked nominal used as a temporal adjunct remains `NOUN`, with a temporal dependency relation (`obl:tmod`). The grammar's term *adverbial* describes its function, not necessarily its UPOS.

| Form / construction | Intended analysis | Status / comment |
|---|---|---|
| `июньышто` ‘in June’ | `NOUN`, `Case=Ine`, `obl:tmod` | **ADOPT**: month name with inessive |
| `шошым` ‘in spring’ | `NOUN`, `Case=Acc`, `obl:tmod` | **ADOPT**: temporal accusative |
| `телым`, `кенгежым`, `шыжым` | `NOUN`, `Case=Acc`, `obl:tmod` | **ADOPT** by same season construction |
| `895-ше ийыште` ‘in the year 895’ | `ийыште`: `NOUN`, `Case=Ine`, `obl:tmod`; year number is adjectival/ordinal | **ADOPT** for noun head; see [adjectives](#3-adjectives-degree-and-adjectival-quantifiers) |
| `шочмын` ‘on Monday’ (and weekdays) | `NOUN Case=Gen` **or** lexicalized `ADV` | **REVIEW**: weekday `-н` described as instructive/adverbial; evidence for lexicalization |
| `эрдене`, `кечывалым`, `кастене`, `йӱдым` ‘in the morning / at noon / in the evening / at night’ | tentative `NOUN` with appropriate case or a postpositional construction with `дене` | **REVIEW**: check segmentation, morphology and lexicalization token by token |
| ‘yesterday’, ‘tomorrow’ | lexical temporal `ADV` | **PROVISIONAL**: unlike transparently inflected temporal nouns |

**Source observation:** names of months use the inessive in temporal adverbials; the four seasons use the accusative; weekday adverbials use a form in `-н`. There is also a note about numeral adverbial `-нек` forms; their POS/features remain to be worked out.

## 3. Adjectives, degree and adjectival quantifiers

### 3.1. Ordinary adjectives and inflection

**ADOPT:** most attributive adjectives do not show many inflectional features. Case/number/possessor morphology is expected especially in apposition, nominalization or ellipsis, where it is actually present.

### 3.2. Short and long forms

**ADOPT:** for the small alternating adjective class (especially colors), annotate **both** `Variant=Short` and `Variant=Long`; absence of `Variant` means the adjective is not known to belong to the alternation.

- **Short:** ordinarily attributive, before a noun.
- **Long:** predicative, appositional, elliptical or nominalized; only the long form takes case, plural, possessive and comparative `-рак` suffixes.
- The same short/long functional opposition is noted for some cardinal numerals and quantifiers (see §4).

### 3.3. Comparison

| Construction | Annotation | Status |
|---|---|---|
| Comparative adjective in `-рак` | `Degree=Cmp` | **ADOPT** |
| Analytic superlative `эн поро` ‘best’ | `эн`: `ADV Degree=Sup`, `advmod` → `поро`; `поро`: `ADJ` with no superlative feature | **ADOPT** |

The adjective *поро* may itself be an `amod` dependent of a following noun. The superlative is encoded on the **separate degree word**, not on the unchanged adjective.

### 3.4. Ordinals and quantifiers

- **ADOPT:** ordinals such as `икымше` → `ADJ` (normally with `NumType=Ord` where used); an ordinal specifying a year is also adjectival.
- **REVIEW:** initial adjective list: `шуко`, `ятыр`, `шагал`, `икмыняр(е)`, `южо`, `иктаж-мыняр`. Some later notes analyze `икмыняр(е)` and `южо` as `DET/PRON` with `PronType=Ind`. Resolve these conflicting POS policies by syntactic use rather than treating all as invariably `ADJ`.

## 4. Numerals

### 4.1. Cardinals and ordinals

- **ADOPT:** cardinal `NUM` examples: `ик`, `кок`, `кум`, etc.; `NumType=Card`.
- **ADOPT:** annotate `Variant=Short|Long` for the numeral paradigms that show this contrast. Long forms may additionally bear `Case`.
- **ADOPT:** ordinary ordinals are treated as `ADJ` in §3 (`NumType=Ord`).

### 4.2. Collective / set numerals

**PROVISIONAL:** `NumType=Sets` for traditional collective numerals formed from long cardinal forms (starting at ‘two’) and some indefinite quantifiers/pronouns by an adverbial/instructive `-н` and an `Ы`-suffix. Example: `коктын` from `кокыт`.

**REVIEW:** check that each such form actually matches UD's ‘sets/collectives’ definition and is not an associative group numeral, general adverb, or another category. Original grammar references: §8.4.1, p. 316; §1.6.4, p. 41.

### 4.3. Group / associative numerals

**ADOPT:** `NumType=Assoc` (Mari-specific value, also used in Erzya) for specialized forms denoting a group of 2–5 persons: ‘the two/three/four/five of us/you/them’.

- Bound bases: `когынь-` ‘two’, `кумынь-` ‘three’, `нылынь-` ‘four’, `визынь-` ‘five’.
- Obligatory **plural possessive suffix**.
- Limited distribution with postpositions or genitive, dative and accusative case marking.
- Do not confuse `Assoc` (‘the N of us’) with `Sets` (‘N sets/pairs of X’).

## 5. Pronouns, determiners and pro-adverbs

### 5.1. General features and possession

| Category | Policy |
|---|---|
| Personal pronouns | `PRON`, `PronType=Prs`, `Person`, `Number` and actual case (`Nom`, `Acc`, `Gen`, `Dat`) |
| Polite `те` | **CONTEXT:** add `Polite=Form` **only when formal address is unmistakable**; otherwise leave unset. No comparable tag in the chosen Erzya/Moksha and Russian datasets. |
| Genitive personal pronoun | **ADOPT:** do not use `Poss=Yes`; use `Case=Gen`, with `nmod:poss` for syntactic possession and `[psor]` on the possessed word |
| Reflexive `шке` | **ADOPT:** `Reflex=Yes`; case and person where actually marked |

Reflexive notes: `шке` may be marked for person and case, but often is not. `шкалан` (dative) and `шкем` (accusative) are examples without person marking; especially frequent in impersonal sentences, but also possible in personal sentences. Consult the paradigm in the grammar.

### 5.2. Demonstratives (`PronType=Dem`)

- Candidate forms: `тиде`, `тудо`, `нине`, `нуно`; also `седе`, `саде`.
- **REVIEW:** `тудо` can be personal (`PronType=Prs`) rather than demonstrative, depending on use.
- `саде` is relatively uncommon as an independent demonstrative and often anaphoric. Derived `садлан` ‘therefore’ and `сандене` ‘therefore, consequently’ have lexicalized meanings.
- **ADOPT, where genuinely lexicalized:** descendants of demonstratives may be independent `ADJ` or `ADV`, rather than retaining the source word's `PronType=Dem` merely on etymological grounds.

### 5.3. Interrogatives (`PronType=Int`)

**ADOPT:** choose UPOS by synchronic category (`ADV`, `ADJ`, `DET`, `PRON`), not by the traditional cover term *interrogative pronoun*.

| Form | Meaning | UPOS | Relevant FEATS |
|---|---|---|---|
| `кушто` | where (inessive) | `ADV` | `PronType=Int` |
| `кушко` / `куш(ко)` | to where (illative) | `ADV` | `PronType=Int` |
| `кушан` | where (lative) | `ADV` | `PronType=Int` |
| `кушечын` / `кушеч(ын)` | from where | `ADV` | `PronType=Int` |
| `кушкен` | for how long; how far | `ADV` | `PronType=Int` |
| `кунам` | when | `ADV` | `PronType=Int` |
| `кунамлан` | till/by when | `ADV` | `PronType=Int` |
| `кунамсек` | since when | `ADV` | `PronType=Int` |
| `кунамсе` | of what time | `ADJ` | `PronType=Int` |
| `кусо` | from where; of what place | `ADJ` | `PronType=Int` |
| `кузе` | how | `ADV` | `PronType=Int` |
| `могай` | what; what kind of | `ADJ` | `PronType=Int` |
| `кунар(е)`, `мыняр(е)` | how many/much | `DET` attributively; `PRON` independently | `PronType=Int` |
| `кунарын`, `мынярын` | ‘in a group of how many’ | `PRON` | `Case=Gen\|PronType=Int` |
| `мынярылан` | at what time / for what price / by what amount | `PRON` | `Case=Dat\|PronType=Int` |
| `мынярымше` | which one; which number | `ADJ` | `NumType=Ord\|PronType=Int` |

Additional interrogatives listed in the grammar but without settled individual UPOS decisions: `кӧ` ‘who’, `мо` ‘what’, `мочол(о)` ‘(oh) how many/much’, `кудо` ‘which’, `молан` ‘why’. **REVIEW** them when they occur.

**CONTEXT:** genuine interrogative vs. relative uses should be distinguished (`PronType=Int` vs `PronType=Rel` as appropriate). Embedded questions can remain interrogative. Several listed forms also function as particles: `мо`, `могай`, `кунам`, `кузе` (grammar §9.5.46, 9.5.48, 9.5.39, 9.5.38); see §10.

### 5.4. Negative pro-forms (`PronType=Neg`)

**ADOPT:** the productive negative series formed with `ни-` from interrogative pro-forms receives `PronType=Neg` (with `PRON`, `ADV`, `DET` or `ADJ` as context warrants), **not** `Polarity=Neg` as well.

- Exception mentioned in the grammar: `иктат` ‘nobody, not one’; its analysis deserves individual review.
- Unlike Mari, the current Erzya-JR and Moksha-JR feature inventories do not provide direct `PronType=Neg` transfer examples.

### 5.5. Reciprocal and reflexive pro-forms

- **ADOPT:** `икте-весе` ‘each other’ → **one** `PRON` token, `PronType=Rcp` (and overt case/possessor features if present). Do not split at the hyphen in the ordinary grammaticalized reciprocal construction. Erzya/Moksha have comparable single-token reciprocal forms.
- **ADOPT:** `шке` → `PRON` with `Reflex=Yes` (see §5.1).

### 5.6. Indefinite and total pro-forms

**ADOPT:** `PronType=Ind` = unspecified individuals, amounts, subsets, times, etc.; `PronType=Tot` = ‘each/every/all/everyone’. Initial morphological sources of indefiniteness noted: `кеч-`, `керек-`, `ала-`, `иктаж-`, `-гынат`; `ик(те)` and `вес(е)` may also function as indefinite `PRON` (**CONTEXT**).

| Form | UPOS | `PronType` | Note/status |
|---|---|---|---|
| `икмыняр(е)` | `DET` / independent `PRON` | `Ind` | Some, a few, several; see POS conflict in §3 |
| `иктаж(е)` | `PRON` / `DET` | `Ind` | Somebody, something, some |
| `иктажышт` | `PRON` | `Ind` | One of them |
| `кажне` | `DET` | `Tot` | Each, every |
| `чумыр` | `DET` / `ADJ` | `Tot` / `_` | **EXPERT QUESTION E2:** total quantifier vs lexical ‘whole/common/general’ adjective |
| `чыла` | `PRON` / `DET` | `Tot` | All, everyone |
| `чылажат` | `ADV` | `_` | **PROVISIONAL:** summative ‘in total, altogether’; reconsider after examples |
| `чылаже` | `ADV` | `_` | **PROVISIONAL:** summative ‘in total, altogether’ |
| `чылажлан` | `PRON` | `Tot` | ‘For everything’; `Case=Dat` |
| `чылан` | `PRON` | `Tot` | Collective ‘all/everyone’ |
| `чыланже` | `ADV` | `_` | **PROVISIONAL:** ‘in all, together’ |
| `чылашт` | `PRON` | `Tot` | All of them |
| `южгунам` | `ADV` | `Ind` | Sometimes, occasionally |
| `южгунамже` | `ADV` | `Ind` | Only sometimes |
| `южикте` | `PRON` | `Ind` | Some (of them) |
| `южиктыже` | `PRON` | `Ind` | One of them |
| `южиктыжышт` | `PRON` | `Ind` | Some of them |
| `южиктышт` | `PRON` | `Ind` | Some of them |
| `южо` | `DET` / `PRON` | `Ind` | Some; typically attributive |
| `южыжо` | `PRON` | `Ind` | Some (of them) |
| `южышт` | `PRON` | `Ind` | Some (of them) |

**EXPERT QUESTION E2 example contrasts:** `чумыр класс` ‘the whole class’; `чумыр йылме` ‘common language’; `чумыр положений` ‘general situation’; `пашам чумыр ышташ` ‘to work as a team’.

## 6. Adpositions

**ADOPT:** Mari postpositions generally receive `UPOS=ADP`, with `AdpType=Post`, regardless of their possible nominal origin. `AdpType=Post` is widely used in the main Uralic transfer languages.

- Include `Case` **when it is an actual inflection on the postposition**, not merely the case that the postposition governs.
- Include applicable `[psor]` features when overtly present.
- Comparative postpositions can have `Degree=Cmp`: `ӱстел деке` ‘to the table’ → `ӱстел декырак` ‘closer to the table’.
- **REVIEW** segmentation and inflection for individual postpositional families and opaque/lexicalized forms.

## 7. Finite verbs, tense, negation and existential predicates

### 7.1. Finite verbal categories

| Feature | Pilot values / notes |
|---|---|
| `Person` | `1`, `2`, `3` |
| `Number` | `Sing`, `Plur` |
| `Mood` | `Ind`, `Imp`, `Des` (indicative, imperative, desiderative) |
| `Tense` | `Pres`, `Past` (two synthetic past paradigms) |
| `VerbForm` | `Fin` for morphologically finite verbs |
| `Polarity` | `Neg` on negative verbal elements; no systematic `Polarity=Pos` |
| `Connegative` | `Yes` on the connegative lexical verb |
| `Evident` | `Nfh` on synthetic past II **as a deliberate pilot convention**, despite semantic reservations |

### 7.2. Simple and compound tenses

| Construction | Morphological annotation | Status |
|---|---|---|
| Simple present | Finite `VERB`, `Tense=Pres` | **ADOPT** |
| Analytic future | lexical infinitive `VERB VerbForm=Inf` + finite future auxiliary (e.g. `тӱҥалаш`) with morphological `Tense=Pres` | **ADOPT**, context-dependent lexical `VERB` vs `AUX` |
| Simple past I | finite `VERB`, `Mood=Ind\|Tense=Past\|VerbForm=Fin` + inflected person/number | **ADOPT** |
| Simple past II (`-ын` paradigm) | finite `VERB`, `Evident=Nfh\|Mood=Ind\|Tense=Past\|VerbForm=Fin` + person/number | **PROVISIONAL:** retain `Evident=Nfh` for pilot; may be removed after review |
| Compound past I | present finite lexical verb + `ыле` (`AUX Tense=Past`) | **ADOPT** |
| Compound past II (second periphrastic imperfect) | present finite lexical verb + `улмаш` (`AUX Evident=Nfh\|Tense=Past`) | **ADOPT** working scheme |
| Compound past III (first periphrastic perfect) | past II finite verb + `ыле` | **REVIEW** interaction with `Evident=Nfh` on component |
| Compound past IV (second periphrastic perfect) | past II finite verb + `улмаш` | **REVIEW** interaction with `Evident=Nfh` on component |

Illustrative feature strings in the notes were taken from different person/number forms: past I `...Number=Sing|Person=3...`, past II `...Number=Sing|Person=1...`. These are **not** restrictions on the corresponding paradigms.

### 7.3. Negation and connegative

**ADOPT:** in a negative auxiliary construction, mark the *negative auxiliary* `Polarity=Neg` (e.g. `ом`) and mark the dependent lexical connegative `Connegative=Yes`; do not put clausal polarity on the lexical connegative merely because the clause is negative. `Polarity=Pos` is not systematically annotated.

### 7.4. Existential уло, улыт, уке

**PROVISIONAL:** distinguish positive existential verbal forms from invariant negative existential *уке*.

| Form/use | Pilot UPOS/FEATS | Dependency policy |
|---|---|---|
| `уло` ‘there is’ | `VERB`, usual 3SG present features; lemma `улаш` | existential predicate head |
| `улыт` ‘there are’ | `VERB`, usual 3PL present features; lemma `улаш` | existential predicate head |
| `уке` ‘there is not’ | `ADJ Polarity=Neg` | nonverbal predicate head |
| `уке улыт` ‘there are not’ | `уке ADJ Polarity=Neg` + `улыт AUX` (finite features) | `улыт` → `уке` with `cop` |

Example working analysis:

```text
Йолташ-влак уке улыт.
'There are no friends.'

Йолташ-влак   NOUN   Number=Plur    nsubj → уке
уке           ADJ    Polarity=Neg   root
улыт          AUX    (finite FEATS) cop → уке
```

**EXPERT QUESTION E1:** Is *уке* synchronically a nonverbal negative existential predicate (adjective-like) or a suppletive negative form of *улаш*? The current `ADJ` decision is tentative. Standalone reply *уке* ‘no’ may instead be a particle; see §10.

## 8. Non-finite verb forms

**ADOPT:** describe the synchronically productive non-finite form, not its historical source. E.g. a synthetic past-II verb originating in a gerund is `VerbForm=Fin`, not `VerbForm=Ger`.

| Type | Example | Pilot FEATS | Status |
|---|---|---|---|
| Infinitive | `кочкаш` ‘to eat’ | `VerbForm=Inf` | **ADOPT** |
| Necessitive infinitive | `кочман` ‘(one) must eat’ | `InfForm=Nec\|VerbForm=Inf` | **ADOPT**; language-specific `InfForm=Nec` |
| Active participle | `кочшо` ‘eating’ | `VerbForm=Part\|Voice=Act` | **ADOPT** for this paradigm |
| Passive participle | `кочмо` ‘eaten’ | `VerbForm=Part\|Voice=Pass` | **PROVISIONAL** where interpretation is not transparently passive |
| Future-necessitive participle | `кочшаш` ‘to eat; to be eaten’ | `PartForm=FutNec\|VerbForm=Part` | **ADOPT** pending **EXPERT QUESTION E3**; alternative `Aspect=Prosp\|VerbForm=Part` |
| Negative participle | `кочдымо` ‘not eating / not eaten’ | `Polarity=Neg\|VerbForm=Part` | **ADOPT** |
| General affirmative/instructive converb | `кочкын` ‘eating’ | `ConvType=Gen\|VerbForm=Conv` | **ADOPT** |
| General negative converb | `кочде` ‘not eating’ | `ConvType=Gen\|Polarity=Neg\|VerbForm=Conv` | **ADOPT** |
| Anterior (‘after’) converb | `кочмек(е)` ‘after eating’ | `ConvType=Ant\|VerbForm=Conv` | **ADOPT** |
| Posterior (‘before’) converb | `кочмеш(ке)` ‘before eating’ | `ConvType=Post\|VerbForm=Conv` | **ADOPT** (also ‘instead of’) |
| Simultaneous (‘while’) converb | `кочшыла` ‘while eating’ | `ConvType=Sim\|VerbForm=Conv` | **ADOPT** |

### 8.1. Language-specific form features

- `InfForm=Nec`: necessitive infinitive.
- `PartForm=FutNec`: future-necessitive participial paradigm.
- `ConvType=Gen|Ant|Post|Sim`: general-purpose, anterior, posterior and simultaneous converb **subparadigms**. `Post` describes the conventional posterior/before subtype; it does **not** guarantee that a following event actually occurs (e.g. ‘instead of’ use).

**ADOPT:** use `ConvType` instead of assigning `Tense=Past|Fut|Pres` purely from the temporal relationship conveyed by `-мек(е)`, `-меш(ке)` and `-шыла`. This distinguishes `кочмешке` from general `кочкын` without confusing relative ordering with morphological tense.

**EXPERT QUESTION E3:** for *-шаш*, is prospectivity consistently the defining grammatical property, or is deontic necessity equally essential? Current pilot preserves the subtype as `PartForm=FutNec`, without pre-committing to `Aspect=Prosp` or `Mood=Nec`.

**REVIEW:** *кочмо* can have active-like semantic uses despite being called a passive participle. Decide whether `Voice=Pass` is a stable paradigm label or must vary with argument structure (potentially prefer a `PartForm` distinction).

## 9. Auxiliaries and multi-verb constructions

### 9.1. Default AUX features and relations

**ADOPT:** grammaticalized `AUX` receives morphologically expressed `Person`, `Number`, `Mood`, `Tense`, `VerbForm` etc. Do **not** introduce `Aspect` systematically merely from a broad lexical meaning; the current pilot uses `aux:aspect` for grammaticalized aspectual verbs.

| Construction | Pilot analysis |
|---|---|
| Negative predicate | negative `AUX` has `Polarity=Neg`; lexical connegative has `Connegative=Yes` |
| Analytic future with `тӱҥалаш` | inflected `AUX` (usually present morphology) attached with `aux` to lexical infinitive |
| Compound past with `ыле` | `AUX Tense=Past` attached with `aux` |
| Compound past with `улмаш` | `AUX Evident=Nfh\|Tense=Past` |
| Grammaticalized aspectual converb construction | lexical converb `VERB ConvType=Gen\|VerbForm=Conv` heads; finite `AUX` uses `aux:aspect` |
| Lexical verb + genuine adjunct converb | do **not** automatically use `AUX`; investigate `advcl`, `xcomp`, etc. |

**CONTEXT:** when *ыле* functions as a copula instead of an invariant compound-tense marker, working notes prefer fuller verbal features:

```text
ыле    улаш    AUX    Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
```

and syntactic `cop`, not `aux`. The grammar traditionally calls *ыле* and *улмаш* particles; their **AUX** analysis is a UD-specific choice.

### 9.2. Other modal / lexical-source auxiliaries listed in the grammar

These are **candidates for `AUX` in grammaticalized contexts**, not unconditional lexeme-level UPOS assignments.

| Lemma | Lexical meaning → grammaticalized meaning |
|---|---|
| `керташ (-ам)` | ‘to be able to’ |
| `мошташ (-ем)` | ‘to be able to’ |
| `лияш (-ям)` | ‘to be, to become’ → ‘to be allowed’ |
| `толашаш (-ем)` | ‘to try, to strive’ → improper/inadequate execution |
| `тӧчаш (-ем)` | ‘to try, to attempt’ → improper/inadequate execution |
| `кӱлаш (-ам)` | ‘to have to’ |
| `верешташ (-ам)` | ‘to get, to catch’ → ‘to have to’ |
| `логалаш (-ам)` | ‘to get to, to find oneself somewhere’ → ‘to have to’ |
| `возаш (-ам)` | ‘to lie down’ → ‘to have to’ |
| `пернаш (-ем)` | ‘to strike against’ → ‘to have to’ |
| `тӱкнаш (-ем)` | ‘to touch (intransitive)’ → ‘to have to’ |

### 9.3. Aspect auxiliaries: working lexical inventory

**ADOPT when truly grammaticalized:** auxiliary `UPOS=AUX`, `aux:aspect` → lexical converb head; normal finite verbal features. **REVIEW:** precise `Aspect` feature values are deferred and no aspect-to-lemma mapping is imposed yet.

| Lemma | Contribution recorded in notes |
|---|---|
| `возаш (-ам)` | resultative; added property; lying position |
| `илаш (-ем)` | continuous |
| `каяш (-ем)` | ingressive; tangible result |
| `кияш (-ем)` | durative; lying position |
| `кодаш (-ам)` | resultative; evident result; excessive |
| `кодаш (-ем)` | resultative; subject leaves |
| `колташ (-ем)` | inchoative; momentary |
| `кошташ (-ам)` | durative; many locations |
| `кудалташ (-ем)` | resultative; impetuous, sudden |
| `кышкаш (-ем)` | resultative; impetuous, sudden; scattering |
| `лекташ (-ам)` | delimitative; unexpectedness |
| `лукташ (-ам)` | exhaustive; creation of something |
| `мияш (-ем)` | gradual |
| `налаш (-ам)` | delimitative; autobenefactive |
| `ончаш (-ем)` | momentary; activity is tried |
| `опташ (-ем)` | iterative; large scale |
| `пуаш (-эм)` | resultative; benefactive |
| `пытараш (-ем)` | exhaustive |
| `пыташ (-ем)` | exhaustive; all participate |
| `пышташ (-ем)` | resultative; lying position; in advance |
| `сеҥаш (-ем)` | exhaustive; ‘to manage to’ |
| `темаш (-ам)` | exhaustive; sufficient |
| `толаш (-ам)` | gradual |
| `шинчаш (-ам)` | resultative; sitting position |
| `шинчаш (-ем)` | durative; sitting position |
| `шогалаш (-ам)` | resultative; standing position |
| `шогалташ (-ем)` | resultative; standing position |
| `шогаш (-ем)` | durative; standing position |
| `шуаш (-ам)` | exhaustive; attainment of a goal |
| `шуаш (-эм)` | resultative; energetic, impetuous, sudden |
| `шукташ (-ем)` | exhaustive; ‘to manage to’ |
| `шындаш (-ем)` | resultative; sitting position |
| `эртараш (-ем)` | delimitative |

The above is an **inventory of possible aspectual uses**. In particular, do not confuse `эртараш` in the aspect-auxiliary list with lexical directional `эрташ` ‘pass by’ in §11.4.

### 9.4. Benefactive / autobenefactive auxiliaries

| Lemma | Contribution |
|---|---|
| `пуаш (-эм)` | benefactive; resultative |
| `налаш (-ам)` | autobenefactive; delimitative |
| `ончыкташ (-ем)` | benefactive (demonstrating); lexical ‘to begin’ |

**REVIEW:** set explicit dependency and feature guidelines for benefactive constructions after examining contextual examples; the raw notes do not settle these.

## 10. Particles

### 10.1. Adopted and tentative cases

- **ADOPT:** `айда`, `айста` (invitation/exhortation, ‘let's/come on’) → `PART`, optionally `PartType=Mod`; in clausal use attach with `discourse`. The inflected main verb keeps its own morphological `Mood` (often indicative).
- **PROVISIONAL:** `тек` (‘let/may’) → `PART` candidate; confirm its subtype and dependency.
- **CONTEXT:** standalone *уке* ‘no’ can be a particle, distinct from negative existential predicate *уке* (§7.4).
- **CONTEXT:** traditional “particles” *ыле*, *улмаш* are `AUX` in periphrastic tense constructions (§9), not automatically `PART`.

### 10.2. Reference-only inventory from the grammar

> **Warning:** these are semantic groupings in the grammar, **not** a verified UD `PART` lexicon. The same surface form can appear in multiple categories and may receive different UPOS in different contexts.

**Demonstrative:** `теве` ‘here, look here’; `тевыс` ‘here, look here’; `уна` ‘there, look there’.

**Limiting / specifying:** `веле` ‘only, just’; `гына` ‘only, just’; `иже` ‘not until’; `керек` ‘at least’; `кеч` ‘at least’; `лач(ак)` ‘only, exclusively’; `тевак` ‘almost, nearly’; `теве` ‘almost, nearly’; `тевыс` ‘almost, nearly’; `улыжат` ‘only, no more than’; `чуч` ‘almost, nearly’; `чылт` ‘right, just’; `ялт` ‘right, just’.

**Intensifying / emphatic:** `вара` ‘well, then’; `веле` ‘more and more’; `вет` ‘so, indeed, well’; `гын` ‘as for …’; `гына` ‘rather, fairly’; `да` ‘really’; `дык` ‘well, so’; `ито` ‘even’; `кеч` ‘if only’; `лаваса` ‘certainly’; `ман` ‘how about that’; `просто` ‘just, simply’; `садак` ‘after all’; `садыгак` ‘after all’; `то-то` ‘there you are’; `тудо` ‘well, of course’; `тушто` ‘there, here’; `уж` ‘indeed’; `уже` ‘already’; `чай` ‘probably’; `э/эй` ‘oh, how’; `эсогыл` ‘even’; `эре` ‘more and more’; `эртак` ‘more and more’.

**Approximate quantity:** `ик` ‘roughly, about’; `иктаж` ‘roughly, about’; `лиеш` ‘soon’ (gloss as in notes).

**Additive:** `адак` ‘still, yet, more’; `да монь` ‘and the like’; `да мо да` ‘and the like’; `монь` ‘and the like’.

**Interrogative:** `а` ‘so, huh’; `ала` ‘maybe, probably, then’; `але` ‘really’; `але вара` ‘really, indeed’; `вет` ‘isn't it’; `гала` ‘really’; `кузе` ‘what, come again’; `кунам` ‘how, when’; `ма` ‘are/is you/…’; `мо` ‘are/is you/…’; `мо вара` ‘then’; `моли` ‘perhaps’; `неужели` ‘really’; `ну` ‘is that so’; `разве` ‘then’.

**Affirmative / confirming:** `ане` ‘yes; so, well’; `да` ‘yes’; `йӧра` ‘fine, okay’; `кернак(ак)` ‘absolutely’; `конешне` ‘of course’; `лаваса` ‘sure’; `лач(ак)` ‘just’; `но` ‘yes, uh-huh’; `ну` ‘really’; `-огыла` ‘well then’; `пеш йӧра` ‘very good’; `товат(ат)` ‘for sure’; `товро` ‘certainly’; `туге` ‘yes’; `туге вет` ‘indeed’; `туге шол` ‘that's how it is’; `тыгак` ‘yes, really’; `тыге` ‘yes, really’; `чынак(ак)` ‘indeed’; `чынжым(ак)` ‘indeed’; `шол` ‘indeed’.

**Doubt / epistemic:** `ала` ‘could be’; `ала-ала` ‘hardly’; `ала мо` ‘possibly’; `аман` ‘apparently’; `векат` ‘probably’; `витне` ‘apparently’; `гай` ‘it seems’; `докан` ‘it seems’; `дыр` ‘perhaps’; `коеш` ‘it seems’; `кузе` ‘how …’; `мо` ‘really’; `можыч` ‘maybe’; `наверне` ‘probably’; `очыни` ‘probably’; `пожале` ‘most likely’; `ужамат` ‘apparently’; `ужат` ‘apparently’; `чай` ‘apparently’; `шонет` ‘it seems’.

**Denial:** `уке` ‘no’; `уке шол` ‘no (emphatic)’.

**Comparison:** `веле` ‘like, as though’; `пуйто` ‘it's like’; `чылт` ‘just like’; `ялт` ‘quite like’.

**Exclamative:** `кузе` ‘how’; `дык` ‘well, so’; `могай` ‘what kind of, how’.

**Indifference:** `керек` ‘makes no difference’; `садак` ‘makes no difference’; `садыгак` ‘makes no difference’.

**Supposition:** `лиеш` ‘if …’; `лийже` ‘let's imagine’.

**Negative:** `-мочет` ‘forgetting about …’; `-моэт` ‘forgetting about …’; `огыл` ‘not’.

**Other:** `икманаш` ‘in a word’; `-можо` ‘it's not about …’; `мутлан` ‘for example’; `на` ‘okay, here you are’; `ну` ‘so, well’; `пожалуйста` ‘please’; `поро лий(за)` ‘please’; `тугат` ‘even without that’; `тугеже` ‘that means’.

## 11. Dependency relations

### 11.1. General inventory for the pilot

The original notes treat the following standard relations as available; individual Mari constructions will be defined after more manual annotation.

```text
nsubj      obj        iobj       csubj      ccomp      xcomp
obl        vocative   expl       dislocated advcl      advmod
discourse  aux        cop        mark       nmod       appos
nummod     acl        amod       det        clf        case
conj       cc         fixed      flat       list       parataxis
compound   orphan     goeswith   reparandum punct      root
dep
```

**Do not use a relation merely because it exists in the inventory.** In particular, a manner converb in a single motion event is not automatically `csubj`.

### 11.2. Specific subrelations

| Relation | Pilot use | Status |
|---|---|---|
| `aux:aspect` | grammaticalized aspectual auxiliary → lexical converb head | **ADOPT** |
| `obl:tmod` | nominal temporal adjunct (`июньышто`, `шошым`, `ийыште`, etc.) | **ADOPT** |
| `compound:svc` | single-event multi-verb construction, especially a manner + directional verb | **PROVISIONAL**; select head convention consistently |
| `nmod:poss` | genitive nominal/pronominal possessor | **ADOPT** |

### 11.3. AUX, copula, particles

- In grammaticalized tense/aspect constructions the **lexical predicate** is the syntactic head even when it is morphologically non-finite; auxiliaries attach with `aux` or `aux:aspect`.
- In a nonverbal predicate with a copula (e.g. provisional `уке улыт`), the nonverbal predicate is head, and `улыт` receives `cop`.
- Invitation particle `айда/айста` can attach to the predicate with `discourse`.
- Postposition receives `case` and attaches to its nominal complement.

### 11.4. Directional converb constructions (`compound:svc`)

**PROVISIONAL:** *йоген эрта* in `Сола гоч Оръю эҥер йоген эрта.` (‘The Orju River flows through Sola’) is a **manner-of-motion + path** complex describing one event, not a clausal subject.

Working decision from the discussion:

- `йоген` (`йогаш`) — lexical `VERB`, `VerbForm=Conv`, **head**.
- `эрта` (`эрташ`, not `эртараш`) — finite directional `VERB`, dependent via **`compound:svc`**.
- Do not label `йоген` `csubj`; do not automatically make directional `эрта` an aspectual `AUX`.
- **REVIEW:** a finite-path-verb-head analysis is also possible; choose one convention after inspecting other directional pairs.

## 12. Questions and checks

### 12.1. Explicit questions for Юлия (human expert)

- [ ] **E1 — Status of `уке`.** Is the negative existential synchronically an adjective-like nonverbal predicate, or a suppletive negative verb in the `улаш` paradigm? Consequences: `ADJ Polarity=Neg` vs `VERB`, and the structure of `уке улыт`. See [§7.4](#74-existential-уло-улыт-уке).
- [ ] **E2 — `чумыр`: quantifier vs adjective.** Are ‘whole class’ and ‘common language’ distinct syntactic uses requiring `DET PronType=Tot` vs `ADJ`, or one adjective lexeme throughout? Compare the four expressions under [§5.6](#56-indefinite-and-total-pro-forms).
- [ ] **E3 — Future-necessitive `-шаш`.** Is its grammatical invariant prospectivity, deontic necessity, or both? Would a semantic `Aspect=Prosp` be justified in addition to, or instead of, morphological `PartForm=FutNec`? See [§8](#8-non-finite-verb-forms).

### 12.2. Other open annotation and corpus issues (not explicitly assigned to Юлия)

- [ ] **R1 — Temporal lexicalization.** Decide `шочмын` (weekday `-н`), daypart forms with `дене`, and ‘yesterday/tomorrow’ according to synchronic morphology and corpus usage.
- [ ] **R2 — Quantifier POS conflict.** Reconcile the initial `ADJ` list (`шуко`, `ятыр`, `шагал`, `икмыняр(е)`, `южо`, `иктаж-мыняр`) with contextual `DET/PRON` choices.
- [ ] **R3 — Sets vs associative numerals.** Confirm that the conventional collective `-н` forms are properly `NumType=Sets`; keep them separate from `NumType=Assoc` group numerals.
- [ ] **R4 — Verbal evidentiality.** Review `Evident=Nfh` on past II, and whether compound past III/IV create conflicting morphological versus constructional evidence values.
- [ ] **R5 — Passive participle.** Determine whether `Voice=Pass` is always valid for `-мЕ` forms or whether a morphological subtype feature would be safer.
- [ ] **R6 — Postpositions.** Confirm overt case/possessor morphology for each family and avoid marking the case governed on the preceding nominal.
- [ ] **R7 — Aspect auxiliary inventory.** Define criteria for grammaticalized `AUX` vs lexical `VERB`, later consider whether and which `Aspect` values should be assigned; independently treat benefactive constructions.
- [ ] **R8 — Directional complex predicates.** Test `compound:svc` versus other UD analyses and establish a head rule across manner/path pairs.
- [ ] **R9 — Traditional particles.** Audit the §10.2 semantic catalog for genuine UD `PART` vs `ADV`, `PRON`, `AUX`, `INTJ`, etc.
- [ ] **R10 — Orthography/feature proofreading.** Verify `улмаш` vs the `ушмаш` spelling in the raw notes; validate case inventory/suffixes; proofread example spellings and the `-нек` numeral-adverb note before publication.
- [ ] **R11 — Unspecified forms.** Add per-token guidelines for interrogatives `кӧ`, `мо`, `мочол(о)`, `кудо`, `молан`, and for `иктат` (negative pro-form without `ни-`).
- [ ] **R12 — Politeness evidence.** Determine whether the corpus contains unambiguous formal-address instances of `те`; otherwise leave `Polite` absent.

## 13. Feature inventory and references

### 13.1. Working morphological feature index

| Domain | Features / values currently envisaged |
|---|---|
| Nominal | `Case=Nom\|Acc\|Gen\|Dat\|Ine\|Ill\|Lat\|Cmp\|Com`; `Number=Sing\|Plur`; `Person[psor]=1\|2\|3`; `Number[psor]=Sing\|Plur` |
| Degree/variants | `Degree=Cmp\|Sup`; `Variant=Short\|Long` |
| Numerals | `NumType=Card\|Ord\|Sets\|Assoc` |
| Pro-forms | `PronType=Prs\|Dem\|Int\|Rel\|Rcp\|Neg\|Ind\|Tot`; `Polite=Form`; `Reflex=Yes` |
| Adpositions | `AdpType=Post`; optionally `Case`, `[psor]`, `Degree=Cmp` |
| Finite verbs | `Person=1\|2\|3`; `Number=Sing\|Plur`; `Mood=Ind\|Imp\|Des`; `Tense=Pres\|Past`; `VerbForm=Fin`; `Connegative=Yes`; `Polarity=Neg`; provisional `Evident=Nfh` |
| Non-finite verbs | `VerbForm=Inf\|Part\|Conv`; `Voice=Act\|Pass`; `InfForm=Nec`; `PartForm=FutNec`; `ConvType=Gen\|Ant\|Post\|Sim`; `Polarity=Neg` |
| Particles | optionally `PartType=Mod` |

This is a **working inventory**, not a validator-complete inventory of all features/values that may appear in Mari data.

### 13.2. Main documentation / source pointers kept from the notes

- [Universal Dependencies documentation](https://universaldependencies.org/)
- [UD `Person[psor]` example documentation](https://universaldependencies.org/hy/feat/Person-psor.html)
- [UD universal morphological features](https://universaldependencies.org/u/feat/index.html)
- [UD dependency relations](https://universaldependencies.org/u/dep/index.html)
- Riese, Bradley & Yefremova, *Mari: An Essential Grammar for International Learners* (grammar references scattered throughout the original notes: numerals §1.6.4 and §8.4.1; non-finite forms §7.2; particles §9.5). Check edition/page numbering in the source being used.
- Comparative annotation references used during discussion: UD Erzya-JR, Moksha-JR, Finnish-TDT, Hungarian-Szeged, Komi-Zyrian; the individual comparisons here are **contextual precedents**, not mandatory Mari policies.

---

*This file is a reorganized working draft for manual annotation. Decisions remain editable; marked expert questions and review checks should not be silently resolved during automated annotation.*

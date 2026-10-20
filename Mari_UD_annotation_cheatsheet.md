# Meadow Mari UD: annotation cheat sheet (pilot)

**Status:** working notes for manual annotation; **not** a frozen UD specification. Organized from the project's 19 October 2026 notes. The proposed Mari-specific features and disputed analyses have not been independently validated. `—` means no feature assigned for this phenomenon, not necessarily `FEATS=_` for the entire token.

**How to use:** (1) choose UPOS from the word's *current function*, (2) select the relevant FEATS under that UPOS, (3) check [context-sensitive cases](#2-context-sensitive-and-unresolved-cases), and (4) assign dependencies using [syntax](#3-syntax). Feature names appear **alphabetically within each UPOS**. Values in example strings are in UD notation (`Number=Sing`, not `Sg`).

## 1. UPOS → FEATS

### `ADJ` — adjective

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Case` | Actual case on inflected/nominalized or elliptical forms | Most ordinary attributive adjectives are uninflected. |
| `Degree` | `Cmp` | Comparative `-рак`; do **not** place `Degree=Sup` on an adjective just because it follows `эн`. |
| `Number` | `Sing`, `Plur` when morphologically expressed | Especially nominalization, ellipsis or apposition. |
| `Number[psor]` | `Sing`, `Plur` if expressed | Inflected long adjectives. |
| `NumType` | `Ord` | Ordinal `икымше`, including ordinals modifying `ий` ‘year’. |
| `Person[psor]` | `1`, `2`, `3` if expressed | Inflected long adjectives. |
| `PronType` | `Int`, `Neg`; perhaps `Tot` where adjectival | `могай` ‘what kind’, `кунамсе`, `кусо`; negative pro-adjectives. |
| `Variant` | `Short`, `Long` | Both marked **only** for alternating adjectives, especially colors: `ош` / `ошо`. `_` for adjectives outside that class. |

**Pilot:** short forms are ordinarily attributive; long forms are predicative, independent, appositional or nominalized and can bear nominal inflection. **Review:** `чумыр` (`ADJ` vs `DET`), and some quantitative modifiers listed in §2.

### `ADP` — adposition

| Feature | Values / use | Examples or notes |
|---|---|---|
| `AdpType` | `Post` | Mari postpositions, e.g. `вошт`, `вач`, `коклаште`. |
| `Case` | Actual case morphology **on the ADP**, not the case it governs | Possible `коклаште` `Case=Ine`; confirm transparently inflected families. |
| `Degree` | `Cmp` | `деке` → `декырак` ‘closer to’. |
| `Number[psor]` | `Sing`, `Plur` where expressed | Some postpositions accept possessive morphology. |
| `Person[psor]` | `1`, `2`, `3` where expressed | Same. |

**Syntax:** ADP normally depends on its nominal complement as `case`. Nominal origin does not automatically make a postposition `NOUN`.

### `ADV` — adverb

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Degree` | `Cmp`, `Sup` where expressed | `эн` in `эн поро` → `ADV Degree=Sup`; `advmod(поро, эн)`. |
| `PronType` | `Int`, `Rel`, `Neg`, `Ind`, `Tot` when genuinely pronominal | `кушто`, `кузе`, `кунам` → `Int`; `нигунам` → `Neg`; `южгунам` → `Ind`. `Tot` only for genuine total pro-adverbs. |

**No automatic `ADV` for a noun used temporally:** `шошым`, `июньышто`, `ийыште` remain `NOUN` + `obl:tmod`. Lexicalized temporal words may be `ADV` (see §2).

**Interrogative pro-adverbs:** `кушто`, `кушко`, `кушан`, `кушечын`, `кушкен`, `кунам`, `кунамлан`, `кунамсек`, `кузе`. **Lexicalized summative adverbs**, provisionally without `PronType`: `чылаже`, `чылажат`, `чыланже`.

### `AUX` — auxiliary

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Evident` | `Nfh` **when adopted for the paradigm** | `улмаш` in compound past II/IV: `Evident=Nfh\|Tense=Past`. |
| `Mood` | `Ind`, `Imp`, `Des` if inflected | Ordinary finite morphological features on grammaticalized auxiliaries. |
| `Number` | `Sing`, `Plur` if inflected | Negative verb, future/aspect auxiliaries, copula. |
| `Person` | `1`, `2`, `3` if inflected | E.g. negative `ом` first singular. |
| `Polarity` | `Neg` | Negative auxiliary such as `ом`; do not annotate `Pos` systematically. |
| `Tense` | `Pres`, `Past` where morphologically appropriate | `тӱҥалам` in analytic future: `Tense=Pres`; compound-past `ыле`: `Tense=Past`. |
| `VerbForm` | `Fin` for synchronically finite forms | May be omitted on **invariant grammaticalized** `ыле` in compound tenses (pilot policy). |

**Pilot construction rules:**

- Analytic future: lexical infinitive head (`VERB VerbForm=Inf`) + grammaticalized `тӱҥалаш` (`AUX`, finite **present** morphology, relation `aux`). A lexical ‘begin’ use remains `VERB` + complement, often `xcomp`.
- Compound past: `ыле` → `AUX Tense=Past`; `улмаш` → `AUX Evident=Nfh|Tense=Past`. If `ыле` is an actual copula, fuller `Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin` and `cop` are proposed.
- Aspectual converb complex: inflected grammaticalized second verb → `AUX` attached by `aux:aspect` to the preceding lexical converb. **No systematic `Aspect` value yet**; normal inflectional FEATS remain.
- Negative predicate: negative `AUX Polarity=Neg`; its lexical connegative is `VERB Connegative=Yes`.

### `DET` — determiner / attributive pro-quantifier

| Feature | Values / use | Examples or notes |
|---|---|---|
| `PronType` | `Dem`, `Int`, `Ind`, `Neg`, `Tot` as applicable | Attributive `мыняр(е)` (how many) → `Int`; `икмыняр(е)`, `южо` → `Ind`; `кажне`, `чыла` → `Tot`. |

**Context:** the same form can be `PRON` when it replaces a noun. `чумыр` as `DET PronType=Tot` is provisional and must be compared with lexical `ADJ` uses.

### `NOUN` — noun

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Case` | `Nom`, `Acc`, `Gen`, `Dat`, `Ine`, `Ill`, `Lat`, `Cmp`, `Com` | Notes: `-м`, `-н`, `-лан`, `-ште`, `-ш(ке)`, `-еш`, `-ла`, `-ге` respectively; **verify suffix–case mapping before publication**. |
| `Number` | `Sing`, `Plur` | Plural markers noted: `-влак`, `-шамыч`, `-ла`, `-мыт`. |
| `Number[psor]` | `Sing`, `Plur` | Number of possessor where overt. |
| `Person[psor]` | `1`, `2`, `3` | Person of possessor where overt. |
| `Polarity` | `Neg` | Verb nominalizations with -дымаш. |
| `VerbForm` | `Vnoun` | Verb nominalizations with -маш and -дымаш. |

**Temporal nominal examples:** `июньышто` → `Case=Ine`, `шошым` → `Case=Acc`, `895-ше ийыште` → `ийыште Case=Ine`, all with `obl:tmod` when modifying a predicate. The same applies to the other seasons (`телым`, `кенгежым`, `шыжым`).

### `NUM` — numeral

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Case` | Actual case on an inflected numeral | Especially long forms and special group numerals. |
| `Number[psor]` | `Plur` in associative/group forms | Bound stems `когынь-`, `кумынь-`, `нылынь-`, `визынь-` require plural possessive morphology. |
| `NumType` | `Card`, `Sets` **(review)**, `Assoc` **(Mari-specific)** | `ик`, `кок`, `кум` = `Card`; `коктын` proposed `Sets` (not yet checked); `когыньна` ≈ ‘the two of us’ = `Assoc`. |
| `Person[psor]` | `1`, `2`, `3` in associative/group forms | E.g. groups of us/you/them. |
| `Variant` | `Short`, `Long` | Mark both members of cardinal short/long paradigm; long forms may inflect. |

**Do not conflate:** `Sets` (‘sets/pairs of X’) with `Assoc` (‘N of us/you/them’). The grammatical class traditionally called *collective numerals* needs review before `Sets` is frozen. Ordinals are currently `ADJ NumType=Ord`, not `NUM`.

### `PART` — particle

| Feature | Values / use | Examples or notes |
|---|---|---|
| `PartType` | `Mod` **(optional)** | Hortative `айда`, `айста` → `PART PartType=Mod`, with `discourse`. |
| `Polarity` | `Neg` **(candidate; verify context)** | Independent negative particle `огыл` if analyzed as a clause negator; not a blanket rule for all traditional particles. |

Other candidate particles: `тек` (hortative), and items in the grammar's long demonstrative, limiting, emphatic, interrogative, affirmative, epistemic and negative particle lists. **Their grammar label does not guarantee `UPOS=PART`.** See §2.

### `PRON` — pronoun

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Case` | `Nom`, `Acc`, `Gen`, `Dat` and other attested inflections | Personal `мыйын`, `тыйын` are genitive personal pronouns; `мынярын Case=Gen`, `мынярылан Case=Dat`. |
| `Number` | `Sing`, `Plur` | Morphological number (`те` stays plural, including polite singular address). |
| `Number[psor]` | `Sing`, `Plur` when overt | Possessive-marked reciprocal/reflexive/other pronouns where applicable. |
| `Person` | `1`, `2`, `3` | Personal pronouns `мый`, `тый`, `те` etc. |
| `Person[psor]` | `1`, `2`, `3` when overt | Mark only real possessive morphology. |
| `Polite` | `Form` only when context unambiguously indicates polite address | `те` or its case forms; otherwise omit. |
| `PronType` | `Prs`, `Dem`, `Int`, `Rel`, `Rcp`, `Neg`, `Ind`, `Tot` | `икте-весе` → `Rcp` (one hyphenated token); `ни-` series → `Neg`; `чыла` → `Tot`; `иктаж` → `Ind`. |
| `Reflex` | `Yes` | `шке`; forms like `шкем`, `шкалан` need not express person. |

**No `Poss=Yes`** just because a personal pronoun is genitive. Use `Case=Gen`, `nmod:poss` on the dependent pronoun in a possessive nominal, and `[psor]` features on the possessed noun. **`PronType=Neg`, not `Polarity=Neg`**, on negative pro-forms.

### `PROPN` — proper noun

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Case` | Actual inflected case, if expressed | E.g. a name in a postpositional construction. |
| `Number` | `Sing`, `Plur` where appropriate | No special Mari PROPN policy yet. |

### `VERB` — lexical verb

| Feature | Values / use | Examples or notes |
|---|---|---|
| `Aspect` | **Not systematically assigned in pilot** | `Aspect=Prosp` on `-шаш` is an alternative to be discussed with Юлия. |
| `Connegative` | `Yes` | Lexical form licensed by negative auxiliary, e.g. `тол` in `ом тол`. |
| `ConvType` | `Gen`, `Ant`, `Post`, `Sim` **(Mari-specific)** | `кочкын` / `кочде` = general; `кочмеке` = anterior; `кочмешке` = posterior/before; `кочшыла` = simultaneous. |
| `Evident` | `Nfh` **(pilot, provisional)** | Synthetic past II (e.g. `тунемынам`); may be removed if too semantically strong. |
| `InfForm` | `Nec` **(Mari-specific)** | `кочман` → `InfForm=Nec|VerbForm=Inf`. |
| `Mood` | `Ind`, `Imp`, `Des` on finite forms | `Mood=Ind` on finite indicative even when used hortatively. |
| `Number` | `Sing`, `Plur` on inflected forms | Finite agreement. |
| `Number[psor]` | `Sing`, `Plur` when overt | Possessive suffixes on some non-finite forms. |
| `PartForm` | `FutNec` **(Mari-specific)** | `кочшаш` → `PartForm=FutNec|VerbForm=Part`; see expert question. |
| `Person` | `1`, `2`, `3` on inflected forms | Finite agreement. |
| `Person[psor]` | `1`, `2`, `3` when overt | Some possessive-marked non-finite forms. |
| `Polarity` | `Neg` when part of the verb's morphology | `кочдымо` (negative participle), `кочде` (negative converb); **not** the lexical connegative following separate negation. |
| `Tense` | `Pres`, `Past` on finite forms | Both synthetic past I/II = `Past`; no `Fut` simply from analytic future or relative temporal meaning of a converb. |
| `VerbForm` | `Fin`, `Inf`, `Part`, `Conv` | Non-finite forms listed below. A historical gerund serving as finite past II is `Fin`, not `Ger`. |
| `Voice` | `Act`, `Pass` on participles (provisional) | `кочшо` = `Act`; `кочмо` = `Pass` by paradigm, but not all uses are semantically passive. |

**Non-finite lookup** (all are `VERB` when used as productive verbal forms):

| Form | FEATS |
|---|---|
| `кочкаш` ‘to eat’ | `VerbForm=Inf` |
| `кочман` ‘must eat’ | `InfForm=Nec|VerbForm=Inf` |
| `кочшо` ‘eating’ | `VerbForm=Part|Voice=Act` |
| `кочмо` ‘eaten’ | `VerbForm=Part|Voice=Pass` **(review)** |
| `кочшаш` ‘to eat; to be eaten’ | `PartForm=FutNec|VerbForm=Part` **(expert)** |
| `кочдымо` ‘not eating/eaten’ | `Polarity=Neg|VerbForm=Part` |
| `кочкын` ‘eating’ | `ConvType=Gen|VerbForm=Conv` |
| `кочде` ‘not eating’ | `ConvType=Gen|Polarity=Neg|VerbForm=Conv` |
| `кочмек(е)` ‘after eating’ | `ConvType=Ant|VerbForm=Conv` |
| `кочмеш(ке)` ‘before eating; instead of eating’ | `ConvType=Post|VerbForm=Conv` |
| `кочшыла` ‘while eating’ | `ConvType=Sim|VerbForm=Conv` |

**Finite tense lookup:** simple present `Tense=Pres`; simple past I `Tense=Past`; simple past II `Evident=Nfh|Tense=Past` **(provisional)**; compound past I `Pres + ыле`; II `Pres + улмаш`; III `Past II + ыле`; IV `Past II + улмаш`. In compound III/IV, the relationship between component-level evidentiality and construction meaning is unresolved.

---

## 2. Context-sensitive and unresolved cases

### 2.1. UPOS decisions that require context

| Word(s) / construction | Candidate analyses | Disambiguation to check |
|---|---|---|
| `уке` | `ADJ Polarity=Neg` as existential predicate; `PART`/`INTJ` as independent ‘no’; perhaps `VERB` if suppletive | Predicate `Юмо уке`; `уке улыт` with `улыт AUX cop`; **ask Юлия** whether adjective-like or verbal. |
| `уло`, `улыт`, finite `улаш` | `VERB` existential; `AUX` copula | Standalone existential predicate vs copular support of a separate nonverbal predicate. |
| `ыле`, `улмаш` | `AUX` in compound tenses; traditional grammar calls them particles | `ыле` also copular `AUX` with `cop` and potentially fuller finite features. **Verify form `улмаш` (notes once have `ушмаш`).** |
| `тӱҥалаш` | `AUX` analytic future; `VERB` ‘begin’ | Compare future `кошташ тӱҥалам` vs lexical inceptive + infinitive (`xcomp`). |
| 33 aspectual verbs (§2.2) | `AUX` with `aux:aspect`; `VERB` when lexical | Has it lost independent event/argument structure? Do **not** assign `Aspect` automatically. |
| Directional verbs, e.g. `эрташ` in `йоген эрта` | second `VERB` in `compound:svc`; ordinary `VERB` elsewhere; possible `AUX` hypothesis | Single manner+path event, not `csubj`; **pilot** `йоген` = head, `эрта` = `VERB compound:svc`. |
| `айда`, `айста`; `тек` | `PART` for hortative use (vs discourse interjection in some contexts) | `PartType=Mod` optional; `discourse` in sample `Айста марла муралтена!`. |
| `чумыр` | `DET PronType=Tot` vs `ADJ` | ‘whole class’ vs ‘common language’; **ask Юлия**. |
| `икмыняр(е)`, `южо`, `иктаж-мыняр`, `шуко`, `ятыр`, `шагал` | Original `ADJ` quantifier list vs `DET`/`PRON` | Attributive vs standalone/pronominal distribution; original notes conflict. |
| `иктаж(е)`, `чыла`, `мыняр(е)`, `кунар(е)` | `DET` attributively; `PRON` independently | Use syntactic distribution, not the grammar's broad “pronoun” designation. |
| `тиде`, `тудо`, `нине`, `нуно`, `саде`, `седе` | `PRON`/`DET` `PronType=Dem`; `тудо` possibly `PronType=Prs` | Demonstrative vs anaphoric/personal reference; derivatives such as `садлан`, `сандене` may be lexical `ADV`. |
| `мо`, `могай`, `кунам`, `кузе` | `PRON`/`ADJ`/`ADV` pro-form vs `PART` | Interrogative, relative, or lexical particle function; `PronType=Int` vs `Rel` when relevant. |
| `кунамсе`, `кусо` | `ADJ PronType=Int` in current pilot | Check attributive uses, versus spatial/temporal pro-adverbs (`кунам`, `кушечын`). |
| `шочмын` (and other weekdays) | `NOUN` with case vs lexical `ADV` | `-н` weekday adverbial ending; nominal genitive vs synchronically adverb-forming suffix. |
| `эрдене`, `кечывалым`, `кастене`, `йӱдым` | Case-marked `NOUN` / postpositional sequence vs lexical `ADV` | Recheck morphology and productivity; do not assume one analysis for all four. |
| `чылаже`, `чылажат`, `чыланже` | `ADV` without `PronType` vs `ADV PronType=Tot` | Summative ‘in total/all together’ vs productive pronominal totality. |
| `иктат` | `PRON PronType=Neg` vs negative-polarity indefinite | Unlike productive `ни-` series. |
| Ordinary `шке` vs its declined/possessive forms | `PRON Reflex=Yes` with form-specific case/person | `шкем`, `шкалан` may lack person marking. |
| Traditional particle inventory (§2.3) | `PART`, `ADV`, `PRON`, `AUX`, `INTJ`, etc. | Context-sensitive: same form appears in multiple semantic particle categories. |

### 2.2. Aspectual auxiliary candidates (lexical uses remain `VERB`)

The grammar lists **33 lemma entries** (including some homonymous or differing inflectional patterns) used in grammaticalized converb complexes. Keep these as a *lookup list*, not a requirement that every occurrence is `AUX`:

`возаш`, `илаш`, `каяш`, `кияш`, `кодаш` (two paradigms), `колташ`, `кошташ`, `кудалташ`, `кышкаш`, `лекташ`, `лукташ`, `мияш`, `налаш`, `ончаш`, `опташ`, `пуаш`, `пытараш`, `пыташ`, `пышташ`, `сеҥаш`, `темаш`, `толаш`, `шинчаш` (two paradigms), `шогалаш`, `шогалташ`, `шогаш`, `шуаш` (two paradigms), `шукташ`, `шындаш`, `эртараш`.

**Other modal/necessity auxiliary candidates:** `керташ`, `мошташ`, `лияш`, `толашаш`, `тӧчаш`, `кӱлаш`, `верешташ`, `логалаш`, `возаш`, `пернаш`, `тӱкнаш`.

**Benefactive/autobenefactive candidates:** `пуаш`, `налаш`, `ончыкташ`.

### 2.3. Common potential particles (not automatically `PART`)

- Hortative: `айда`, `айста`, `тек`.
- Limiting/emphatic: `веле`, `гына`, `кеч`, `лач(ак)`, `вет`, `садак`, `чай`, etc.
- Interrogative/response: `мо`, `кузе`, `кунам`, `але`, `ну`, `да`, `йӧра`, etc.
- Negation/denial: `огыл`, `уке` (‘no’ as reply).

The long semantic particle inventory remains in [the full project guidelines](Mari_UD_annotation_guidelines.md); these are only frequent candidate families.

### 2.4. Expert questions and remaining reviews

**For Юлия (explicitly noted in the source):**

- [ ] **E1 — `уке`:** synchronically adjective-like negative existential or suppletive member of the `улаш` verb paradigm? This controls `ADJ` vs `VERB` and `уке улыт` structure.
- [ ] **E2 — `чумыр`:** is `DET PronType=Tot` (‘whole’) syntactically distinct from `ADJ` (‘common/general’)? Compare `чумыр класс` and `чумыр йылме`.
- [ ] **E3 — `-шаш` participle:** does its invariant semantics privilege prospectivity (`Aspect=Prosp`) or deontic necessity, or both? Pilot uses `PartForm=FutNec` to avoid deciding prematurely.

**Annotation / corpus checks (not yet assigned to Юлия):**

- [ ] Verify `Voice=Pass` on the `-ме` participle in active-like interpretations; perhaps use a form subclass instead.
- [ ] Check `NumType=Sets` on collective `-н` numerals (`коктын`) separately from group `NumType=Assoc`.
- [ ] Inspect weekday/daypart lexicalization and the modifier UPOS of quantifiers.
- [ ] Verify postpositional case suffixes and the noun case inventory; do not confuse postposition inflection with governed case.
- [ ] Review whether `Evident=Nfh` overstates past-II semantics; inspect compound past III/IV.
- [ ] Validate `compound:svc` and the head choice for manner + directional-verb complexes.
- [ ] Confirm form `улмаш` vs accidental `ушмаш`; proofread paradigms and spellings before committing as official policy.

---

## 3. Syntax

**Reference:** [UD v2 — Universal Dependency Relations](https://universaldependencies.org/u/dep/) (**37 universal relations**). The compact English descriptions below paraphrase UD's general definitions; they are **not** intended to replace the individual UD documentation pages. Syntax rules beneath the main table are Mari-specific **pilot decisions**.

### 3.1. Universal relations (alphabetical)

| Relation | Short definition |
|---|---|
| [`acl`](https://universaldependencies.org/u/dep/acl.html) | Clause modifying a nominal (including finite/nonfinite and relative clauses). |
| [`advcl`](https://universaldependencies.org/u/dep/advcl.html) | Adverbial subordinate clause modifying a predicate (time, condition, purpose, etc.). |
| [`advmod`](https://universaldependencies.org/u/dep/advmod.html) | Adverbial or other nonclausal modifier of a predicate/modifier. |
| [`amod`](https://universaldependencies.org/u/dep/amod.html) | Adjectival modifier of a nominal. |
| [`appos`](https://universaldependencies.org/u/dep/appos.html) | Appositional nominal providing an alternative description/identification. |
| [`aux`](https://universaldependencies.org/u/dep/aux_.html) | Auxiliary contributing grammatical information to lexical predicate. |
| [`case`](https://universaldependencies.org/u/dep/case.html) | Adposition or other case marker dependent on nominal complement. |
| [`cc`](https://universaldependencies.org/u/dep/cc.html) | Coordinating conjunction attached to its conjunct. |
| [`ccomp`](https://universaldependencies.org/u/dep/ccomp.html) | Clausal complement with its own subject or without obligatory control. |
| [`clf`](https://universaldependencies.org/u/dep/clf.html) | Classifier modifying nominal/numeral expression. |
| [`compound`](https://universaldependencies.org/u/dep/compound.html) | Compound constituents behaving as a lexical unit. |
| [`conj`](https://universaldependencies.org/u/dep/conj.html) | Nonfirst member of a coordination. |
| [`cop`](https://universaldependencies.org/u/dep/cop.html) | Copula supporting a nonverbal predicate. |
| [`csubj`](https://universaldependencies.org/u/dep/csubj.html) | Clausal subject of a predicate. |
| [`dep`](https://universaldependencies.org/u/dep/dep.html) | Unspecified dependency; last resort when no relation fits. |
| [`det`](https://universaldependencies.org/u/dep/det.html) | Determiner of a nominal. |
| [`discourse`](https://universaldependencies.org/u/dep/discourse.html) | Discourse/pragmatic item, interjection or particle. |
| [`dislocated`](https://universaldependencies.org/u/dep/dislocated.html) | Element displaced from its canonical clausal position. |
| [`expl`](https://universaldependencies.org/u/dep/expl.html) | Expletive or pleonastic nominal (e.g., dummy pronoun). |
| [`fixed`](https://universaldependencies.org/u/dep/fixed.html) | Fixed grammaticalized multiword expression. |
| [`flat`](https://universaldependencies.org/u/dep/flat.html) | Head-flat structure such as multipart names without internal dependency syntax. |
| [`goeswith`](https://universaldependencies.org/u/dep/goeswith.html) | Orthographically split parts of what should be one word. |
| [`iobj`](https://universaldependencies.org/u/dep/iobj.html) | Indirect object in a language's core-object distinction. |
| [`list`](https://universaldependencies.org/u/dep/list.html) | Elements of loosely structured lists (e.g. addresses). |
| [`mark`](https://universaldependencies.org/u/dep/mark.html) | Subordinating marker of a clause. |
| [`nmod`](https://universaldependencies.org/u/dep/nmod.html) | Nominal modifier of a nominal head. |
| [`nsubj`](https://universaldependencies.org/u/dep/nsubj.html) | Nominal subject of a predicate. |
| [`nummod`](https://universaldependencies.org/u/dep/nummod.html) | Numeric modifier of a nominal. |
| [`obj`](https://universaldependencies.org/u/dep/obj.html) | Core direct object of a predicate. |
| [`obl`](https://universaldependencies.org/u/dep/obl.html) | Oblique nominal modifier/argument of a predicate (including place/time). |
| [`orphan`](https://universaldependencies.org/u/dep/orphan.html) | Residual dependent in a gapping/ellipsis construction. |
| [`parataxis`](https://universaldependencies.org/u/dep/parataxis.html) | Loosely connected clause/expression without ordinary subordination. |
| [`punct`](https://universaldependencies.org/u/dep/punct.html) | Punctuation attached to the appropriate syntactic head. |
| [`reparandum`](https://universaldependencies.org/u/dep/reparandum.html) | Word/phrase superseded by a repair in disfluent speech. |
| [`root`](https://universaldependencies.org/u/dep/root.html) | Root of a dependency tree (head index `0`). |
| [`vocative`](https://universaldependencies.org/u/dep/vocative.html) | Direct address term outside ordinary argument structure. |
| [`xcomp`](https://universaldependencies.org/u/dep/xcomp.html) | Open predicative/clausal complement whose subject is obligatorily controlled externally. |

### 3.2. Mari-specific subtypes and constructions

| Relation | Pilot use / example | Status |
|---|---|---|
| `aux:aspect` | Grammaticalized aspectual second verb → lexical converb; e.g. `йӧратен шындем` (the converb heads). | **Adopted locally**; this is a non-universal subtype of `aux`. |
| [`compound:svc`](https://universaldependencies.org/u/dep/compound-svc.html) | Single motion event from converb manner + finite path verb; `compound:svc(йоген, эрта)` in `йоген эрта` ‘flows past’. | **Provisional**; review the head choice. |
| [`nmod:poss`](https://universaldependencies.org/u/dep/nmod-poss.html) | Genitive possessor modifies a noun; `nmod:poss(пӧртет, тыйын)` ‘your house’. | Working convention. |
| [`obl:tmod`](https://universaldependencies.org/u/dep/obl-tmod.html) | Temporal **nominal** adjunct; `шошым`, `июньышто`, `ийыште`. | Working convention. |
| `nmod:obj` | Direct object or nominalised verbs with -маш. | Working convention. |

Other general UD subtypes may become appropriate (`acl:relcl`, `nsubj:pass`, etc.) after the first manually verified sample; they are **not** yet independently specified in these Mari notes.

### 3.3. Four useful contrasts

| Construction | Choose | Not |
|---|---|---|
| `ийыште`, `шошым` modifying a predicate | `NOUN` + `obl:tmod` | `ADV` solely because it answers ‘when?’ |
| `айста` in hortative clause | `PART` + `discourse` | `AUX Mood=Imp` merely for exhortative speech act |
| Grammaticalized aspectual paired verb | converb `VERB` head + `AUX aux:aspect` | `csubj` for the converb |
| `йоген эрта` manner + path | **provisional** `йоген` head + `эрта VERB compound:svc` | `csubj` or automatic aspectual `AUX` |

---

## 4. References and scope

- [UD v2: dependency relations](https://universaldependencies.org/u/dep/) — source for relation terminology and short paraphrased definitions.
- [UD universal feature index](https://universaldependencies.org/u/feat/index.html) — standard features and values; Mari-specific subtypes above are explicitly provisional.
- [Full Mari UD annotation guidelines](Mari_UD_annotation_guidelines.md) — longer working notes, auxiliary meanings, and traditional particle catalog.

**Editorial rule:** change a provisional entry only after recording the reason and (ideally) an attested example; especially avoid inferring morphosyntactic features from English translations alone.

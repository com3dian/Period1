[ ] Union (also called disjunction)

```

```

[ ] 

Above is important concept

## Lecture

[ ] Knowledge representation and reasoning

[ ] Reasoning/Understanding/Knowledge

​	-- [ ] Reasoning about family relations

​	-- [ ] Coreference resolution

​	-- [ ] Reasoning about superlatives

​	-- [ ] Time reasoning

[ ] knowleedge base{}

-- [ ] Knowledge representation

-- [ ]

[ ] Probabilitical logic

## Propositional logic

[ ] Propositional logic

-- [ ] proposition {命题}

```
statement? no any statement is a proposition.

```

[ ] Composite propositions

[ ] Syntax

[ ] Model

-- [ ] valuation

	A model (of a logic) is a precisely described situation in which the truth of each formula is determined.
	A model m is a model of a formula α, if α is true in m.

-- [ ] Truth table

-- [ ] Model of a KB

```
M(α) – the set of all models of α, (M(α) = {m | m |= α}).
```



[ ] pragmatics

[ ] Validity

```
A formula α is valid (notation |= α) iff m |= α holds for every
model m
```

[ ] Entailment

```
Entailment is a relationship between sentences (i.e., syntax) that is
based on semantics.
Also called Logical consequence.

```

## FOL

[ ] Fragment of first order logic

-- [ ] Alan Turing

```
There exists no Turing Machine (algorithm) that decides for any
finite set of formulas Γ and any formula α whether Γ |= α.
```

[ ]  Decidable {可判定的}

https://en.wikipedia.org/wiki/Decidability_(logic)

```
Logical system is decidable if membership in the set of logically valid formulas can be effectively determined.

FOL is not decidable, Propositional logic is not decidable
```

[ ] Semi-decidable

```
If a statement is valid, we can find it in finite time (in other words: there
exists a proof we can check for correctness). But there is no effective procedure for checking that a sentence is not valid.
```



## Description Logic



[ ] Formal Ontologies {本体论，本体}

-- [ ] class, individual, relation

```
class also called concepts.
Concepts represent sets of individuals
roles represent binary relations betweennthe individuals
individual names represent single individuals in the domain
```

-- [ ] 

-- [ ] 

[ ] Atomic concept/role/ name{原子化的}

[ ] Boolean constructors

-- [ ] existential restriction/value restriction

``` 
concept language
```

[ ] Semantics/Interpretation

[ ]

[ ] subsumption/assertion

-- [ ] TBox

-- [ ] ABox

# THE LANGUAGE OF FIRST -ORDER LOGIC

[ ] atom

[ ] arity

[ ] predicate symbol/function symbol

[ ] abbreviations



# A Description Logic Primer

[ ] axioms

```
(The DL ontology) consists of a set of statements, called axioms, each of
which must be true in the situation described.
```

-- [ ] assertional (ABox) axioms

Asserting Facts with ABox Axioms.

```
ABox axioms capture knowledge about named individuals
Mother(julia) - julia is mother
Role assertions describe relations between named individuals
parentOf(julia, john) - julia is parent of john
```

-- [ ] terminological (TBox) axioms

```
RBox axioms refer to properties of roles.
Person ≡ Human - Concept equivalence
```

-- [ ] relational (RBox) axioms

```
RBox axioms refer to properties of roles.
parentOf ⊑ ancestorOf - parentOf is a subrole of ancestorOf
```

[ ] top concept

```r
matrix(unlist(OM), ncol =, nrow =)
```
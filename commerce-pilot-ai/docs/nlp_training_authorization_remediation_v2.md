# NLP training authorization remediation V2

This package repairs only findings F-01 through F-05 from the rejected Phase 2C NLP training-authorization review. It does not authorize model training or execution.

V2 uses locale-independent Latin-only lowercase, a separate minimal empty-text stage, and an explicitly ordered duplicate pipeline including hashtag and repeated-punctuation rules. The two source reaudits remain numerically identical to V1.

Experiments A and C are source-native five-class rating classification with labels 1–5, macro F1 primary, and no derived sentiment mapping. Their scores cannot be directly compared across datasets. B2 remains native four-class ASTD sentiment; E remains native binary offensive-language safety.

The V2 authorization lists exact learned configurations rather than a Cartesian grid: A=4, B2=6, C=4, E=6. Dummy/majority remains a non-learned metric floor. AutoML, random/Bayesian search, transformers, neural or embedding models, commercial use, production, protected Test access, and Egyptian-commerce validation remain prohibited.

The rejected package remains historical evidence. V2 is only a candidate and remains `PENDING_INDEPENDENT_REVIEW`.

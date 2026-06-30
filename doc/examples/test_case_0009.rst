.. _test_case_0009:

Large-scale, real-world problem with steady-state data
======================================================

This page describes the model selection problem available in the `PEtab Select test suite </test_suite>`_, case ``0009``.

It reproduces the histone H4
acetylation model selection problem of Blasi *et al.* (2016) [Blasi2016]_.
This involves a model space of :math:`2^{32} \approx 4.3` billion models. The model is formulated
as a ordinary differential equation system based on biochemical reactions. The dataset is real experimental data, measured at steady-state. Briefly, the model selection problem is to identify which subset of 32 parameters should exist in the model to explain the data.

The files for this model selection problem are available at
`test_cases/0009 <https://github.com/PEtab-dev/petab_select/tree/main/test_cases/0009>`_.

Biological background
---------------------

Histones are the proteins around which DNA is wrapped, and histone N-terminal
"tails" have post-translational modifications that regulate
chromatin. The N-terminal tail of histone H4 has four lysine (K) residues that
can each be acetylated: K5, K8, K12, and K16.

Because each of the four sites is either acetylated or not, there are
:math:`2^4 = 16` possible acetylation patterns, called motifs, ranging from
the fully unacetylated state to the fully acetylated state . The
research question in [Blasi2016]_ is to determine how the acetylation patterns are generated: do the acetylation reactions all occur at the same rate regardless of the current acetylation motif, or are some of them motif-specific?

As described below, there are a total of 32 acetylation reactions. The original publication [Blasi2016]_ found that the "best" model had 7 of these reactions that occur at a motif-specific rate, and the rest occur at a shared rate.

The mathematical model
----------------------

The model is a reaction network over the 16 acetylation motifs of the
histone H4 tail. Each motif is a chemical species :math:`x_m`, and a reaction
adds (acetylation) or removes (deacetylation) a single acetyl group, connecting
motifs that differ at exactly one site. Hence, there are 32 acetylation reactions
and 32 deacetylation reactions. The dynamics use mass-action
kinetics, and the measured abundances are assumed to be at steady state
(acetylation/deacetylation are much faster than the cell cycle); in the PEtab
problem this is encoded by measurements at ``time = inf``.

The 16 motifs (state variables)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The state variables ``x_<motif>`` grouped by the number of acetylated lysines (of K5, K8, K12, K16) are:

- zero acetylated lysines: ``x_0ac``
- one acetylated lysine: ``x_k05``, ``x_k08``, ``x_k12``, ``x_k16``
- two acetylated lysines: ``x_k05k08``, ``x_k05k12``, ``x_k05k16``, ``x_k08k12``, ``x_k08k16``, ``x_k12k16``
- three acetylated lysines: ``x_k05k08k12``, ``x_k05k08k16``, ``x_k05k12k16``, ``x_k08k12k16``
- four acetylated lysines: ``x_4ac``

The 32 acetylation reactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each acetylation reaction converts one motif :math:`p` into another motif :math:`q`, :math:`p \to q`, at rate :math:`a_{p\to q}\, a_b\, x_p`,
where :math:`a_b` is the basal (shared) acetylation rate and :math:`a_{p\to q} \equiv` ``a_<p>_<q>`` is a factor that represents the research question. This factor is either fixed to one for reactions that occur with the shared basal rate constant, or the factor is estimated, to change the rate constant to be motif-specific.
Each
acetylation reaction also has an associated deacetylation reaction :math:`q \to p` with rate
:math:`d\, x_q`, where :math:`d \equiv` ``da_b`` is a single, shared basal deacetylation rate constant (fixed to ``1``).

The acetylation reactions are explicitly:

.. code-block:: text

           x_0ac -> x_k05            (rate  a_0ac_k05 * a_b * x_0ac)
           x_0ac -> x_k08            (rate  a_0ac_k08 * a_b * x_0ac)
           x_0ac -> x_k12            (rate  a_0ac_k12 * a_b * x_0ac)
           x_0ac -> x_k16            (rate  a_0ac_k16 * a_b * x_0ac)
           x_k05 -> x_k05k08         (rate  a_k05_k05k08 * a_b * x_k05)
           x_k05 -> x_k05k12         (rate  a_k05_k05k12 * a_b * x_k05)
           x_k05 -> x_k05k16         (rate  a_k05_k05k16 * a_b * x_k05)
           x_k08 -> x_k05k08         (rate  a_k08_k05k08 * a_b * x_k08)
           x_k08 -> x_k08k12         (rate  a_k08_k08k12 * a_b * x_k08)
           x_k08 -> x_k08k16         (rate  a_k08_k08k16 * a_b * x_k08)
           x_k12 -> x_k05k12         (rate  a_k12_k05k12 * a_b * x_k12)
           x_k12 -> x_k08k12         (rate  a_k12_k08k12 * a_b * x_k12)
           x_k12 -> x_k12k16         (rate  a_k12_k12k16 * a_b * x_k12)
           x_k16 -> x_k05k16         (rate  a_k16_k05k16 * a_b * x_k16)
           x_k16 -> x_k08k16         (rate  a_k16_k08k16 * a_b * x_k16)
           x_k16 -> x_k12k16         (rate  a_k16_k12k16 * a_b * x_k16)
        x_k05k08 -> x_k05k08k12      (rate  a_k05k08_k05k08k12 * a_b * x_k05k08)
        x_k05k08 -> x_k05k08k16      (rate  a_k05k08_k05k08k16 * a_b * x_k05k08)
        x_k05k12 -> x_k05k08k12      (rate  a_k05k12_k05k08k12 * a_b * x_k05k12)
        x_k05k12 -> x_k05k12k16      (rate  a_k05k12_k05k12k16 * a_b * x_k05k12)
        x_k05k16 -> x_k05k08k16      (rate  a_k05k16_k05k08k16 * a_b * x_k05k16)
        x_k05k16 -> x_k05k12k16      (rate  a_k05k16_k05k12k16 * a_b * x_k05k16)
        x_k08k12 -> x_k05k08k12      (rate  a_k08k12_k05k08k12 * a_b * x_k08k12)
        x_k08k12 -> x_k08k12k16      (rate  a_k08k12_k08k12k16 * a_b * x_k08k12)
        x_k08k16 -> x_k05k08k16      (rate  a_k08k16_k05k08k16 * a_b * x_k08k16)
        x_k08k16 -> x_k08k12k16      (rate  a_k08k16_k08k12k16 * a_b * x_k08k16)
        x_k12k16 -> x_k05k12k16      (rate  a_k12k16_k05k12k16 * a_b * x_k12k16)
        x_k12k16 -> x_k08k12k16      (rate  a_k12k16_k08k12k16 * a_b * x_k12k16)
     x_k05k08k12 -> x_4ac            (rate  a_k05k08k12_4ac * a_b * x_k05k08k12)
     x_k05k08k16 -> x_4ac            (rate  a_k05k08k16_4ac * a_b * x_k05k08k16)
     x_k05k12k16 -> x_4ac            (rate  a_k05k12k16_4ac * a_b * x_k05k12k16)
     x_k08k12k16 -> x_4ac            (rate  a_k08k12k16_4ac * a_b * x_k08k12k16)

The 32 reverse deacetylations occur at rate e.g. ``da_b * x_k05`` for ``k05 -> 0ac``. These are not listed explicitly but are present for every acetylation reaction above.

The ordinary differential equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on these reactions and rates, the ODE system is therefore

.. math::

   \frac{\mathrm{d}x_{\mathrm{0ac}}}{\mathrm{d}t} &= d\left(x_{\mathrm{k05}} + x_{\mathrm{k08}} + x_{\mathrm{k12}} + x_{\mathrm{k16}}\right) - a_b\left(a_{\mathrm{0ac}\to\mathrm{k05}} + a_{\mathrm{0ac}\to\mathrm{k08}} + a_{\mathrm{0ac}\to\mathrm{k12}} + a_{\mathrm{0ac}\to\mathrm{k16}}\right)x_{\mathrm{0ac}} \\
   \frac{\mathrm{d}x_{\mathrm{k05}}}{\mathrm{d}t} &= a_b\,a_{\mathrm{0ac}\to\mathrm{k05}} x_{\mathrm{0ac}} + d\left(x_{\mathrm{k05k08}} + x_{\mathrm{k05k12}} + x_{\mathrm{k05k16}}\right) - a_b\left(a_{\mathrm{k05}\to\mathrm{k05k08}} + a_{\mathrm{k05}\to\mathrm{k05k12}} + a_{\mathrm{k05}\to\mathrm{k05k16}}\right)x_{\mathrm{k05}} - d\,x_{\mathrm{k05}} \\
   \frac{\mathrm{d}x_{\mathrm{k08}}}{\mathrm{d}t} &= a_b\,a_{\mathrm{0ac}\to\mathrm{k08}} x_{\mathrm{0ac}} + d\left(x_{\mathrm{k05k08}} + x_{\mathrm{k08k12}} + x_{\mathrm{k08k16}}\right) - a_b\left(a_{\mathrm{k08}\to\mathrm{k05k08}} + a_{\mathrm{k08}\to\mathrm{k08k12}} + a_{\mathrm{k08}\to\mathrm{k08k16}}\right)x_{\mathrm{k08}} - d\,x_{\mathrm{k08}} \\
   \frac{\mathrm{d}x_{\mathrm{k12}}}{\mathrm{d}t} &= a_b\,a_{\mathrm{0ac}\to\mathrm{k12}} x_{\mathrm{0ac}} + d\left(x_{\mathrm{k05k12}} + x_{\mathrm{k08k12}} + x_{\mathrm{k12k16}}\right) - a_b\left(a_{\mathrm{k12}\to\mathrm{k05k12}} + a_{\mathrm{k12}\to\mathrm{k08k12}} + a_{\mathrm{k12}\to\mathrm{k12k16}}\right)x_{\mathrm{k12}} - d\,x_{\mathrm{k12}} \\
   \frac{\mathrm{d}x_{\mathrm{k16}}}{\mathrm{d}t} &= a_b\,a_{\mathrm{0ac}\to\mathrm{k16}} x_{\mathrm{0ac}} + d\left(x_{\mathrm{k05k16}} + x_{\mathrm{k08k16}} + x_{\mathrm{k12k16}}\right) - a_b\left(a_{\mathrm{k16}\to\mathrm{k05k16}} + a_{\mathrm{k16}\to\mathrm{k08k16}} + a_{\mathrm{k16}\to\mathrm{k12k16}}\right)x_{\mathrm{k16}} - d\,x_{\mathrm{k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k08}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05}\to\mathrm{k05k08}} x_{\mathrm{k05}} + a_{\mathrm{k08}\to\mathrm{k05k08}} x_{\mathrm{k08}}\right) + d\left(x_{\mathrm{k05k08k12}} + x_{\mathrm{k05k08k16}}\right) - a_b\left(a_{\mathrm{k05k08}\to\mathrm{k05k08k12}} + a_{\mathrm{k05k08}\to\mathrm{k05k08k16}}\right)x_{\mathrm{k05k08}} - 2\,d\,x_{\mathrm{k05k08}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k12}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05}\to\mathrm{k05k12}} x_{\mathrm{k05}} + a_{\mathrm{k12}\to\mathrm{k05k12}} x_{\mathrm{k12}}\right) + d\left(x_{\mathrm{k05k08k12}} + x_{\mathrm{k05k12k16}}\right) - a_b\left(a_{\mathrm{k05k12}\to\mathrm{k05k08k12}} + a_{\mathrm{k05k12}\to\mathrm{k05k12k16}}\right)x_{\mathrm{k05k12}} - 2\,d\,x_{\mathrm{k05k12}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05}\to\mathrm{k05k16}} x_{\mathrm{k05}} + a_{\mathrm{k16}\to\mathrm{k05k16}} x_{\mathrm{k16}}\right) + d\left(x_{\mathrm{k05k08k16}} + x_{\mathrm{k05k12k16}}\right) - a_b\left(a_{\mathrm{k05k16}\to\mathrm{k05k08k16}} + a_{\mathrm{k05k16}\to\mathrm{k05k12k16}}\right)x_{\mathrm{k05k16}} - 2\,d\,x_{\mathrm{k05k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k08k12}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k08}\to\mathrm{k08k12}} x_{\mathrm{k08}} + a_{\mathrm{k12}\to\mathrm{k08k12}} x_{\mathrm{k12}}\right) + d\left(x_{\mathrm{k05k08k12}} + x_{\mathrm{k08k12k16}}\right) - a_b\left(a_{\mathrm{k08k12}\to\mathrm{k05k08k12}} + a_{\mathrm{k08k12}\to\mathrm{k08k12k16}}\right)x_{\mathrm{k08k12}} - 2\,d\,x_{\mathrm{k08k12}} \\
   \frac{\mathrm{d}x_{\mathrm{k08k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k08}\to\mathrm{k08k16}} x_{\mathrm{k08}} + a_{\mathrm{k16}\to\mathrm{k08k16}} x_{\mathrm{k16}}\right) + d\left(x_{\mathrm{k05k08k16}} + x_{\mathrm{k08k12k16}}\right) - a_b\left(a_{\mathrm{k08k16}\to\mathrm{k05k08k16}} + a_{\mathrm{k08k16}\to\mathrm{k08k12k16}}\right)x_{\mathrm{k08k16}} - 2\,d\,x_{\mathrm{k08k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k12k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k12}\to\mathrm{k12k16}} x_{\mathrm{k12}} + a_{\mathrm{k16}\to\mathrm{k12k16}} x_{\mathrm{k16}}\right) + d\left(x_{\mathrm{k05k12k16}} + x_{\mathrm{k08k12k16}}\right) - a_b\left(a_{\mathrm{k12k16}\to\mathrm{k05k12k16}} + a_{\mathrm{k12k16}\to\mathrm{k08k12k16}}\right)x_{\mathrm{k12k16}} - 2\,d\,x_{\mathrm{k12k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k08k12}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05k08}\to\mathrm{k05k08k12}} x_{\mathrm{k05k08}} + a_{\mathrm{k05k12}\to\mathrm{k05k08k12}} x_{\mathrm{k05k12}} + a_{\mathrm{k08k12}\to\mathrm{k05k08k12}} x_{\mathrm{k08k12}}\right) + d\,x_{\mathrm{4ac}} - a_b\,a_{\mathrm{k05k08k12}\to\mathrm{4ac}}x_{\mathrm{k05k08k12}} - 3\,d\,x_{\mathrm{k05k08k12}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k08k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05k08}\to\mathrm{k05k08k16}} x_{\mathrm{k05k08}} + a_{\mathrm{k05k16}\to\mathrm{k05k08k16}} x_{\mathrm{k05k16}} + a_{\mathrm{k08k16}\to\mathrm{k05k08k16}} x_{\mathrm{k08k16}}\right) + d\,x_{\mathrm{4ac}} - a_b\,a_{\mathrm{k05k08k16}\to\mathrm{4ac}}x_{\mathrm{k05k08k16}} - 3\,d\,x_{\mathrm{k05k08k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k05k12k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05k12}\to\mathrm{k05k12k16}} x_{\mathrm{k05k12}} + a_{\mathrm{k05k16}\to\mathrm{k05k12k16}} x_{\mathrm{k05k16}} + a_{\mathrm{k12k16}\to\mathrm{k05k12k16}} x_{\mathrm{k12k16}}\right) + d\,x_{\mathrm{4ac}} - a_b\,a_{\mathrm{k05k12k16}\to\mathrm{4ac}}x_{\mathrm{k05k12k16}} - 3\,d\,x_{\mathrm{k05k12k16}} \\
   \frac{\mathrm{d}x_{\mathrm{k08k12k16}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k08k12}\to\mathrm{k08k12k16}} x_{\mathrm{k08k12}} + a_{\mathrm{k08k16}\to\mathrm{k08k12k16}} x_{\mathrm{k08k16}} + a_{\mathrm{k12k16}\to\mathrm{k08k12k16}} x_{\mathrm{k12k16}}\right) + d\,x_{\mathrm{4ac}} - a_b\,a_{\mathrm{k08k12k16}\to\mathrm{4ac}}x_{\mathrm{k08k12k16}} - 3\,d\,x_{\mathrm{k08k12k16}} \\
   \frac{\mathrm{d}x_{\mathrm{4ac}}}{\mathrm{d}t} &= a_b\left(a_{\mathrm{k05k08k12}\to\mathrm{4ac}} x_{\mathrm{k05k08k12}} + a_{\mathrm{k05k08k16}\to\mathrm{4ac}} x_{\mathrm{k05k08k16}} + a_{\mathrm{k05k12k16}\to\mathrm{4ac}} x_{\mathrm{k05k12k16}} + a_{\mathrm{k08k12k16}\to\mathrm{4ac}} x_{\mathrm{k08k12k16}}\right) - 4\,d\,x_{\mathrm{4ac}}

The observation model
^^^^^^^^^^^^^^^^^^^^^^

Each observable is the steady-state abundance of one motif,

.. math::

   y_m(\theta) = x_m^{\mathrm{ss}}(\theta),

for the 15 observed motifs, :math:`m \in \mathcal{O}`. The observed set :math:`\mathcal{O}` excludes the ``x_k05k16`` motif because it was below the quantification limit of the experiment.
The measured are assumed to follow a log-normal (multiplicative) noise distribution:

.. math::

   \ln \bar{y}_{m,r} = \ln y_m(\theta) + \varepsilon_{m,r},
   \qquad \varepsilon_{m,r} \sim \mathcal{N}(0, \sigma^2),

where :math:`\bar{y}_{m,r}` is replicate :math:`r` of the measurement of motif
:math:`m`, and :math:`\sigma =` ``sigma_`` is the noise parameter (fixed to
``1``).

The likelihood function
^^^^^^^^^^^^^^^^^^^^^^^^^

The likelihood of the estimated parameters :math:`\theta` (the basal rate
:math:`a_b` and the estimated motif-specific factors) is

.. math::

   \mathcal{L}(\theta) = \prod_{m \in \mathcal{O}} \prod_{r=1}^{R_m}
   \frac{1}{\bar{y}_{m,r}\, \sigma \sqrt{2\pi}}
   \exp\!\left( -\frac{\left(\ln \bar{y}_{m,r} - \ln y_m(\theta)\right)^2}{2\sigma^2} \right),

and the corresponding negative log-likelihood, which PEtab Select uses to
compute model selection criteria, is

.. math::

   \mathrm{NLLH}(\theta) = \sum_{m \in \mathcal{O}} \sum_{r=1}^{R_m}
   \left[ \ln\!\left(\bar{y}_{m,r}\, \sigma \sqrt{2\pi}\right)
   + \frac{\left(\ln \bar{y}_{m,r} - \ln y_m(\theta)\right)^2}{2\sigma^2} \right],

where :math:`R_m` is the number of replicates of motif :math:`m`.

Experimental data
-----------------

The model is fitted to the relative abundances of the H4 acetylation motifs
measured in *Drosophila melanogaster* Kc cells. The data come from the
quantitative mass-spectrometry study of Feller *et al.* (2015), as used by
Blasi *et al.* [Blasi2016]_:

Briefly, histone H4 was extracted from wild-type Kc cells. The relative abundances of the H4 N-terminal acetylation motifs were quantified by liquid chromatography–mass spectrometry (LC–MS). The measurements represent the steady-state distribution of motifs.

Of the 16 motifs, one (``K5K16``) lies below the quantification limit and is
not used; hence, the PEtab problem has 15 observables (one per measured motif)
and 252 measurements in total.

The model selection problem
---------------------------

There is one hypothesis per acetylation reaction: its rate is either the shared
basal rate (:math:`a_{p\to q}` fixed to ``1``) or an estimated
motif-specific rate (:math:`a_{p\to q}` estimated). With 32
reactions, the model space therefore contains
:math:`2^{32} \approx 4.3` billion models. The task is to identify the subset of
reactions that require a motif-specific rate constant to explain the data. The published best model
had seven motif-specific reactions [Blasi2016]_.

This PEtab Select formulation differs from the original publication in three
ways in that it omits 11 additional models that were considered in the original publication. These 11 models add negligible computational cost to the model selection problem and were not amongst the best models in the original publication, so are ignored here. Furthermore, we use the FAMoS search method as a general strategy, instead of the highly-tailored problem-specific approach used in the original publication that enabled them to use the brute-force method.

The PEtab Select files
----------------------

PEtab Select problem YAML
^^^^^^^^^^^^^^^^^^^^^^^^^

The problem file specifies the criterion, search method, model space file and additional arguments for the search method.

.. literalinclude:: ../../test_cases/0009/petab_select_problem.yaml
   :language: yaml

The ``candidate_space_arguments`` configure FAMoS:

- ``predecessor_model``: the initial model the search starts from (see below).
- ``critical_parameter_sets``: empty here — no reaction is forced to always be
  motif-specific.
- ``swap_parameter_sets``: a single set containing all 32 reaction parameters.
  FAMoS *lateral* (swap) moves exchange an estimated parameter for a fixed one;
  restricting swaps to within a set means any motif-specific reaction may be
  swapped for any other.
- ``consecutive_laterals: true``: keep performing lateral moves while they keep
  improving the model.
- ``summary_tsv``: where to write a summary of the search status and history.

Model space
^^^^^^^^^^^

The model space is a single subspace ``M`` with 32 parameter columns, one
per acetylation reaction. Every column has the value ``1.0;estimate``, meaning
each reaction can be either fixed to the basal rate (``1.0``) or estimated
(motif-specific). This concisely encodes all ~4.3 billion
models in a single row (the table is wide, with one column per reaction):

.. csv-table::
   :file: ../../test_cases/0009/model_space.tsv
   :delim: tab
   :header-rows: 1

The PEtab problem (``petab/``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model space references a standard PEtab problem that defines the superset
model (all reactions present):

- ``model.xml``: the SBML model with the 16 motif species and the acetylation /
  deacetylation reactions.
- ``parameters.tsv``: the 32 reaction-rate factors ``a_<reaction>`` (``log10``
  scale, bounds ``[1e-3, 1e3]``), the basal acetylation rate ``a_b``, the
  deacetylation reference rate ``da_b`` (fixed to ``1``), and the noise
  parameter ``sigma_``.
- ``observables.tsv``: 15 observables, one per measured motif, and the log-normal noise distribution.
- ``measurements.tsv``: the 252 steady-state (``time = inf``) abundance
  measurements.
- ``conditions.tsv``: a single dummy condition.

Predecessor (initial) model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``predecessor_model.yaml`` is the model the FAMoS search is initialised from, in the PEtab Select model YAML file format. It
is a specific starting model, with 15 motif-specific reactions, that was found
to reproducibly lead to the best model.
Without it, solving the problem from scratch requires on the order of 100
randomly initialised FAMoS searches.

.. literalinclude:: ../../test_cases/0009/predecessor_model.yaml
   :language: yaml

Expected results
^^^^^^^^^^^^^^^^

``expected.yaml`` is the expected selected model, in the PEtab Select model YAML file format. It has the seven
motif-specific reactions of the published best model
(``a_0ac_k08``, ``a_k05_k05k12``, ``a_k12_k05k12``, ``a_k16_k12k16``,
``a_k05k12_k05k08k12``, ``a_k12k16_k08k12k16``, ``a_k08k12k16_4ac``) plus the
basal rate ``a_b`` — eight estimated parameters in total — with
``AICc ≈ -1708.1``.

.. literalinclude:: ../../test_cases/0009/expected.yaml
   :language: yaml

``expected_summary.tsv`` is the FAMoS search trajectory, one row per iteration,
showing how the method switches between ``forward``, ``backward``, and
``lateral`` moves and how the criterion improves at each step:

.. csv-table::
   :file: ../../test_cases/0009/expected_summary.tsv
   :delim: tab
   :header-rows: 1

Why this problem is challenging
-------------------------------

This problem is difficult to solve. The search space is large (~4.3 billion models); hence, many models have criterion values that differ by less than numerical noise, so the
ranking of near-optimal models is effectively non-deterministic across
machines and tolerances. Plain forward or backward selection mostly fails to
reach the optimum.

The FAMoS method reproducibly
converges to models with markedly better criterion values.
Multi-start FAMoS searches recover the published best model while calibrating
only a small fraction (~0.002 %) of the full model space, making this
large-scale problem computationally feasible.

Because of the numerical-noise sensitivity noted above, when running this test
case you should expect to obtain a similar (but not necessarily identical)
``expected_summary.tsv`` (a few rows may be reordered, or the path through model space
may differ). However, the improvement in criterion value over consecutive rows should be conserved, and the same select model should be found.

References
----------

.. [Blasi2016] Blasi T, Feller C, Feigelman J, Hasenauer J, Imhof A, Theis FJ,
   Becker PB, Marr C. *Combinatorial Histone Acetylation Patterns Are Generated
   by Motif-Specific Reactions.* Cell Systems, 2016, 2(1):49–58.
   https://doi.org/10.1016/j.cels.2016.01.002

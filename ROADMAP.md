# Roadmap

GPkit rests on one observation: many of the relationships that govern engineering design are
compatible with geometric programming, so problems that look hard can be solved to a global optimum,
reliably and in seconds. gpkit-core connects that mathematics to the way engineers work, in
subsystems, named variables and physical units.

## What we're working toward

The mathematics underneath GPkit is solid. Our goal is to make the software as solid as the
mathematics. That means a small set of core concepts (models, variables, constraints, vectors,
solutions, and the identifiers that connect them), each with a definition that fits in a sentence,
and code, documentation and file formats that all agree with that definition.

This is the theme because GPkit's bugs and awkward corners keep tracing back to a concept that was
never pinned down. `Model` is both an optimization problem and a reusable set of constraints, so
components carry a meaningless cost (#252). A variable's identifier says where its model falls in a
sequence rather than what the variable is, so inserting a submodel renumbers its neighbors (#368),
and identifiers don't survive saving and reloading a model (#298). Fixing each alone treats a
symptom. We would rather decide what each thing is, once, and let the code follow.

## What that looks like

**Results can be trusted.** GPkit should never be quietly wrong: if it can't give the right answer,
it should say so. Engineers refine a design by re-solving many times, and
that only works if they believe each result.

**The documentation is worth reading.** It is badly out of date and far from where it should be. A
clear, accurate, fun rewrite is central to this work, not cleanup after it: if a concept can't be
explained in a paragraph, it isn't finished, so writing the docs shows us what is still unsettled.
The first step is a published reference for the intermediate representation, the JSON form of a
model and its solution.

## How we decide

The mathematics wins arguments. Objects that represent mathematical ideas should mirror them
exactly, and the machinery around them, such as keys and references, gets the same care.

Real models test the design from the outside: a multi-component example exposes clumsiness that no
unit test will.

Structure stays visible. A signomial constraint gives up the guarantee of a global optimum. Some
problems need one, but reaching for it should be a conscious choice, and the tools should show the
user what forced it.

GPkit is pre-1.0, so we break APIs when the design calls for it and update the example libraries in
the same pass, with no deprecation shims.

Less is more. We bulletproof the few things that must work, and we don't ship a derived quantity
until its math is verified and can be stated in one sentence.

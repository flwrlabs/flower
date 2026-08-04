# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first four.
SPHINXOPTS    ?= -W --keep-going
UV            ?= uv
PYTHON_VERSION ?= 3.11.14
UVRUN         ?= $(UV) run --project .. --locked --python=$(PYTHON_VERSION)
SPHINXBUILD   ?= $(UVRUN) sphinx-build
PYTHON        ?= $(UVRUN) python
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without an argument shows the available targets.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help docs serve Makefile

docs:
	$(MAKE) clean
	$(MAKE) html

serve: docs
	$(PYTHON) -m http.server --directory "$(BUILDDIR)/html"

# Catch-all target: route all unknown targets to Sphinx using make mode.
# $(O) is a shortcut for additional Sphinx options.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
LINKCHECKDIR  = build/linkcheck

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

docs:
	rm -rf build/html
	sphinx-build -D language=en -A lang=True source build/html/

update-text:
	rm -rf build/gettext
	make gettext

update-lang:
	sphinx-intl update -p build/gettext -l $(lang) -w 76

serve:
	make docs
	python -m http.server --directory build/html

checklinks:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(LINKCHECKDIR)"
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

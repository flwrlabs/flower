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
	@test -n "$(lang)" || (echo "Usage: make update-lang lang=<locale>" >&2; exit 1)
	@if [ "$(lang)" = "fr" ]; then \
		command -v msgmerge >/dev/null || (echo "msgmerge is required; install gettext" >&2; exit 1); \
		command -v msginit >/dev/null || (echo "msginit is required; install gettext" >&2; exit 1); \
		mkdir -p locales/$(lang)/LC_MESSAGES; \
		for pot in build/gettext/*.pot; do \
			[ -e "$$pot" ] || { echo "No POT files found in build/gettext" >&2; exit 1; }; \
			domain=$$(basename "$$pot" .pot); \
			po="locales/$(lang)/LC_MESSAGES/$$domain.po"; \
			if [ -f "$$po" ]; then \
				msgmerge --update --backup=off --width=88 "$$po" "$$pot"; \
			else \
				msginit --no-translator --locale="$(lang)" --input="$$pot" --output-file="$$po" --width=88; \
			fi; \
		done; \
	else \
		sphinx-intl update -p build/gettext -l $(lang); \
	fi

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

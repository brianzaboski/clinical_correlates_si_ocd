# verify_firth_logistf.R
#
# Refits the pipeline's in-sample models with established implementations and
# compares the results with the pipeline's output: the Firth logistic
# regressions with logistf (profile penalized-likelihood CIs and tests, logistf
# defaults) and the global likelihood-ratio tests with glm.
#
# Run from the repository root after the pipeline's checks stage (or a full run):
#   Rscript verify_firth_logistf.R
# Requires: install.packages("logistf")

suppressPackageStartupMessages(library(logistf))

out <- "outputs"  # the pipeline's --out folder
d  <- read.csv(file.path(out, "insample_checks_data.csv"))                 # N = 231, as analyzed
dc <- read.csv(file.path(out, "insample_checks_data_complete_cases.csv"))  # complete cases
py <- read.csv(file.path(out, "firth_logistic.csv"))                       # pipeline estimates

feats <- c("age", "female", "medicated", "qol", "psc", "docs", "octcdq_ha", "octcdq_inc",
           "stai_state", "stai_trait", "bfas_n", "bfas_a", "bfas_c", "bfas_e", "bfas_o")

fml <- function(vars) as.formula(paste("si ~", paste(vars, collapse = " + ")))

# logistf's estimates next to the pipeline's, one row per term
compare <- function(fit, label) {
  keep <- names(fit$coefficients) != "(Intercept)"
  term <- names(fit$coefficients)[keep]
  ref  <- py[py$model == label, ]
  ref  <- ref[match(term, ref$term), ]
  data.frame(model = label, term = term,
             OR_R = exp(fit$coefficients[keep]), OR_py = ref$OR,
             lo_R = exp(fit$ci.lower[keep]),     lo_py = ref$OR_lo,
             hi_R = exp(fit$ci.upper[keep]),     hi_py = ref$OR_hi,
             p_R  = fit$prob[keep],              p_py  = ref$p,
             row.names = NULL)
}

res <- rbind(
  do.call(rbind, lapply(c(feats, "bdi_minus9"), function(v)
    compare(logistf(fml(v), data = d), "Univariable"))),
  compare(logistf(fml(feats), data = d), "Features"),
  compare(logistf(fml(c(feats, "bdi_minus9")), data = d), "Features + depression"),
  compare(logistf(fml(feats), data = dc), "Features (complete cases)"),
  compare(logistf(fml(c(feats, "bdi_minus9")), data = dc),
          "Features + depression (complete cases)"))

print(format(res, digits = 3), row.names = FALSE)
cat("\nLargest absolute differences, logistf minus pipeline (expect < 0.001):\n")
cat(sprintf("  OR %.1e   CI lower %.1e   CI upper %.1e   p %.1e\n",
            max(abs(res$OR_R - res$OR_py)), max(abs(res$lo_R - res$lo_py)),
            max(abs(res$hi_R - res$hi_py)), max(abs(res$p_R - res$p_py))))

# Global tests use ordinary (unpenalized) maximum likelihood, so separately
# fitted nested models can be compared directly.
m0 <- glm(si ~ bdi_minus9, family = binomial, data = d)
m1 <- glm(fml(c("bdi_minus9", feats)), family = binomial, data = d)
m2 <- glm(si ~ bdi_minus9 + pc1, family = binomial, data = d)
cat("\nFeatures beyond depression (pipeline: chi2(15) = 33.7, p = .004):\n")
print(anova(m0, m1, test = "LRT"))
cat("\nGeneral-distress component beyond depression (pipeline: chi2(1) = 0.00, p = .96):\n")
print(anova(m0, m2, test = "LRT"))
cat("\nComponent OR per SD beyond depression, Wald CI (pipeline: 0.99 [0.58, 1.67]):\n")
print(round(exp(cbind(OR = coef(m2), confint.default(m2)))["pc1", ], 2))

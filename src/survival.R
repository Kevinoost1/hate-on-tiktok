.libPaths(c("/zfsstore/projects/pi-mai/r_packages", .libPaths()))

library(survival)
library(coxme)
library(readr)
library(dplyr)
library(ggplot2)
library(splines)
library(lme4)

LOCAL_DIR = "/home/kevin/Documents/phd/Project-Data-Donation/survival_results.csv"
ALICE_DIR = "/projects/pi-mai/Project-Data-Donation/survival_results.csv"

load_data <- function(){
  df <- read_csv('survival.csv', show_col_types = FALSE)

  df$topic <- trimws(df$topic)

  df$topic <- factor(df$topic, levels=c('Other', 'News, Politics & Societal Issues'))
  df$user_id <- factor(df$user_id)
  df$high_hate <- factor(df$high_hate, levels=c(0, 1))

  # RD Only select rows as viewed when watch time is more than 3 -> in line with meta's policy for reals
  df <- df %>% filter(df$watch_time_s >= 3)

  return (df)
}

define_event <- function(df){
  df$event <- as.integer(df$watch_time_s <= df$video_duration)
  return (df)
}


ml_cox_randomint <- function(df, type){
  print('start uncorrected model')
  print(type)

  start <- Sys.time()
  formula_uncorrected <- reformulate(c(type, "(1 | user_id)"), response = "Surv(watch_time_s, event)")
  uncorrected_fit <- coxme(formula_uncorrected, data=df)

  print(type)
  print(summary(uncorrected_fit))
  print(paste('ended uncorrected model in', round(as.numeric(difftime(Sys.time(), start, 'mins')), 2)))

  print('start corrected model')
  print(type)

  start <- Sys.time()
  formula_corrected <- reformulate(c(type, "video_duration", "(1 | user_id)"), response = "Surv(watch_time_s, event)")
  corrected_fit <- coxme(formula_corrected, data=df)

  print(summary(corrected_fit))
  print(paste('ended corrected model in', round(as.numeric(difftime(Sys.time(), start, 'mins')), 2)))

  return (list(corrected_fit = corrected_fit, uncorrected_fit = uncorrected_fit))
}

prop_hazard_assumption <- function(df) {
    return (coxph(
      Surv(watch_time_s, event) ~ topic + video_duration + cluster(user_id),data=df
    ) |> 
    cox.zph()
  )
}

write_results <- function(ml_cox_fit){
  results <- rbind(cbind(model = "corrected", as.data.frame(summary(ml_cox_fit$corrected_fit)$coefficients)),
    cbind(model = "uncorrected",as.data.frame(summary(ml_cox_fit$uncorrected_fit)$coefficients)))

  results$CI_lower <- exp(results$coef - 1.96 * results$`se(coef)`)
  results$CI_upper <- exp(results$coef + 1.96 * results$`se(coef)`)
  
  print(results$CI_lower)

  write.csv(results, 'survival_results.csv', row.names = FALSE)
}


# Load and Edit the Data
df <- define_event(load_data())


# Fit the actual Model
ml_cox_fit_topic <- ml_cox_randomint(df, 'topic')
ml_cox_fit_hate <- ml_cox_randomint(df, 'high_hate')
write_results(ml_cox_fit_topic)

print("CI for high hate, quick print dont judge code prettyness lol")
s <- summary(ml_cox_fit_hate$corrected_fit)
ci <- cbind(
  Lower = exp(s$coefficients[, "coef"] - 1.96 * s$coefficients[, "se(coef)"]),
  Upper = exp(s$coefficients[, "coef"] + 1.96 * s$coefficients[, "se(coef)"])
)
print(ci)

# -----------------------------------------------------





# .libPaths(c("/zfsstore/projects/pi-mai/r_packages", .libPaths()))

library(survival)
library(coxme)
library(readr)
library(dplyr)
library(ggplot2)
library(splines)
library(lme4)
library(broom.mixed)

LOCAL_DIR = "/home/kevin/Documents/phd/Project-Data-Donation/hate_source_results.csv"
ALICE_DIR = "/projects/pi-mai/Project-Data-Donation/hate_source_results.csv"


load_data <- function(){
  df <- read_csv('hate_source.csv')

  df <- df %>% mutate(
      user_id = factor(user_id),
      official = factor(official),
      high_hate = as.numeric(high_hate)
    )
  # Unknown is the reference
  df$official <- relevel(df$official, ref = "3")
  return (df)
}


model <- function(df){
  fit <- glmer(high_hate ~ official + (1 | user_id), data = df, family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa"))
  return (fit)
}

construct_results <- function(results){
  results <- tidy(fit, effects = "fixed", conf.int = TRUE) %>%
      mutate(
        odds_ratio = exp(estimate),
        conf_low_OR = exp(conf.low),
        conf_high_OR = exp(conf.high)
      )

  return (results)
}

df <- load_data()
fit <- model(df)

results <- construct_results(fit)
write_csv(results, "hate_source_results.csv")
print(results)

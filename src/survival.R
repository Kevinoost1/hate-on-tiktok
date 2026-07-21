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

COLORS <- c(
  "Other" = "#ec8e3a" ,
  "News, Politics & Societal Issues" = "#001157",
  "Highly Disengaged for News, Politics & Societal Issues (-1 SD)" = "#b03aa3",
  "Average Participant" = "#001157",
  "Highly Engaged for News, Politics & Societal Issues (1 SD)" = "#4F8B4C"
)


load_data <- function(){
  df <- read_csv('survival.csv', show_col_types = FALSE)

  df$topic <- trimws(df$topic)
  df$sent <- trimws(df$sent)

  df$topic <- factor(df$topic, levels=c('Other', 'News, Politics & Societal Issues'))
  df$sent <- factor(df$sent)
  df$user_id <- factor(df$user_id)

  # RD Only select rows as viewed when watch time is more than 3 -> in line with meta's policy for reals
  df <- df %>% filter(df$watch_time_s >= 3)

  return (df)
}

define_event <- function(df){
  df$event <- as.integer(df$watch_time_s <= df$video_duration)
  return (df)
}


ml_cox <- function(df){
  # We made video_duation non-linear -> longer/shorter videos are likely different experiences
  corrected_fit <- coxme(
    Surv(watch_time_s, event) ~ topic + video_duration + (topic | user_id), data=df
  )

  uncorrected_fit <- coxme(
      Surv(watch_time_s, event) ~ topic + (topic | user_id), data=df
    )

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
  
  write.csv(results, 'survival_results.csv', row.names = FALSE)
}


# Load and Edit the Data

df <- define_event(load_data())

# Check Some Assumptions

# print(prop_hazard_assumption(df))
# test_topic_ph_assumption(df)

# Fit the Model

ml_cox_fit <- ml_cox(df)
print(summary(ml_cox_fit$corrected_fit))
print(summary(ml_cox_fit$uncorrected_fit))

write_results(ml_cox_fit)

# -----------------------------------------------------





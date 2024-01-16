library(yaml)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggdist)
library(stringr)
library(hrbrthemes)
library(bbplot)
library(ggthemes)
library(ggthemr)
library(zeallot)

setwd("C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Project")
source('load_data.R')
source('make_plot.R')
source('outlier_check.R')
input_folder <- 'C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\'
output_folder <- 'C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Thesis\\figures\\'
d_input_folder <- 'D:\\Afstuderen\\'

hrbrthemes::import_roboto_condensed()

save_plot <- function(filename) {
  ggsave(paste(output_folder, filename, '.png', sep = ''), width = 7, height = 4, dpi = 300)
}

no_filter <- function(yaml){
  return(TRUE);
}


####################
# BASELINE RESULTS #
####################
baseline_group <- function(yaml, csv) {
  return(-1)
}
baseline_left_filepath <- paste(d_input_folder, 'embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-0\\rnn-0', sep='')
baseline_left <- load_data(baseline_left_filepath, no_filter, baseline_group)

# Baseline Difficult
baseline_difficult_filepath <- paste(input_folder, "baseline\\baseline-difficult", sep='')
baseline_difficult <- load_data(baseline_difficult_filepath, no_filter, baseline_group)

#############################
# PERFECT EMBEDDING RESULTS #
#############################

### Load Perfect Left
perfect_group <- function(yaml, csv) {
  return(0);
}
perfect_left_filepath <- paste(d_input_folder, 'embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-1\\rnn-0', sep='')
perfect_left <- load_data(perfect_left_filepath, no_filter, perfect_group)

plot_regret(perfect_left, c("-1" = "Perfect"), "Perfect Left", 6500, 50000, 0, 4000)

perfect_directions_groups <- c(
  "left_directions" = -2,
  "leftright_directions" = -3,
  "all_directions" = -4
)
perfect_directions_group <- group_by_property(perfect_directions_groups, "env$task_file")
perfect_directions_filepath <- paste(input_folder, "perfect\\perfect-directions", sep='')
perfect_directions_data <- load_data(perfect_directions_filepath, no_filter, perfect_directions_group)

perfect_directions_labels <- c(
  "-2" = "Left",
  "-3" = "Left & Right",
  "-4" = "All"
)

plot_regret(perfect_directions_data, perfect_directions_labels, "Perfect Directions", ymax=20000)

#########################
# EMBEDDING CONSUMPTION #
#########################
consumption_group <- function(yaml, csv) {
  csv$group <- paste(yaml$policy$embedding_obs_init, yaml$policy$embedding_rnn_init)
  return(csv$group)
}
consumption_filepath <- paste(d_input_folder, 'embedding-consumption\\embedding-fifty-logs', sep='')
consumption_data <- load_data(consumption_filepath, no_filter, consumption_group)

# Default cherry-picked view
consumption_overview_data <- data.frame()
for (init in list(list(0, 0), list(0, 1), list(0, 2), list(1, 0), list(2, 0))) {
  temp_data <- consumption_data[consumption_data$group == paste(init[1], init[2]),]
  print(paste("obs-", init[1], "-rnn-", init[2], sep=''))
  #consumption_overview_filepath <- paste(d_input_folder, 'embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-', init[1], '\\rnn-', init[2], '\\', sep='')
  #temp_data <- load_data(consumption_overview_filepath, no_filter, consumption_group)
  consumption_overview_data <- rbind(consumption_overview_data, temp_data)
}
# consumption_overview_data = filter(consumption_data, group == "0 0" | group == "0 1" | group == "0 3" | group == "1 0" | group == "2 0")
consumption_overview_labels <- c("0 0" = "Baseline (no info)", "0 1" = "LSTM Hidden", "0 2" = "LSTM Cell", "2 0" = "State Proxy Concat", "1 0" = "Obs Concat")
plot_regret(consumption_overview_data, consumption_overview_labels, "Embedding Consumption", ymax=8000)
save_plot('embedding-consumption')

plot_regret(consumption_overview_data, consumption_overview_labels, "Embedding Consumption", ymax=70, cumulative=FALSE)
save_plot('embedding-consumption-regret')

# Proof that the uninformed agent converges to the Bayesian-optimal policy
consumption_long_run_data <- filter(consumption_overview_data, group == "0 0")
consumption_long_run_labels <- c("0 0" = "Baseline (no info)")
plot_regret(consumption_long_run_data, consumption_long_run_labels, "Embedding Consumption", 6500, 100000, 0000, 8000)
save_plot('baseline')

plot_regret(consumption_long_run_data, consumption_long_run_labels, "Embedding Consumption", 6500, 100000, 0, 50, FALSE)
save_plot('baseline-regret')

# Generate plots for Appendix
title_labels <- c(
  "No Concat",
  "Observation Concat",
  "State proxy Concat",
  "Observation & State Proxy Concat"
)
for (obs_init in 0:3) {
  obs_init_str <- paste(obs_init, '')
  consumption_obs_data <- filter(consumption_data, str_detect(group, obs_init_str))
  consumption_obs_labels <- setNames(
    c("Blank Init", "Hidden", "Cell", "Hidden & Cell"),
    c(paste(obs_init, 0), paste(obs_init, 1), paste(obs_init, 2), paste(obs_init, 3))
  )
  title <- paste("Embedding Consumption:", title_labels[obs_init + 1])
  plot_regret(consumption_obs_data, consumption_obs_labels, title, ymax=8000)
  save_plot(paste('embedding-consumption-', obs_init, sep = ''))
}

for (obs_init in 0:3) {
  for (rnn_init in 0:3) {
    consumption_regrets <- consumption_data[consumption_data$group == paste(obs_init, rnn_init) & consumption_data$z.env_steps == 50000, "metrics.cumulative_regret_total"]
    print(paste(obs_init, ' ', rnn_init, ' $', round(mean(consumption_regrets), 0), ' \pm ', round(sd(consumption_regrets), 0), '$', sep=''))
  }
}

############################
# SENTENCE EMBEDDING MODEL #
############################
model_embedder_groups <- c(
  "sentences_word2vec_pos" = 1,
  "sentences_word2vec" = 0,
  "sentences_infersent" = 2,
  "sentences_sbert" = 3,
  "sentences_simcse" = 4
)
model_embedder_group <- group_by_property(model_embedder_groups, "env$task_file")
model_direction_groups <- c(
  "one_direction" = 0,
  "all_directions" = 1
)
model_direction_group <- group_by_property(model_direction_groups, "env$task_file")
model_group <- function(yaml, csv) {
  return(paste(model_direction_group(yaml), model_embedder_group(yaml)));
}

model_filepath <- paste(d_input_folder, 'embedding-model\\embedding-model-logs', sep='')
model_data <- load_data(model_filepath, no_filter, model_group)

# Visualize for 0) only left, and 1) left, up, down, right and negation
# TODO finish this once the baseline and perfect are available
for (i in list(0, 1)) {
  model_difficulty_data <- model_data[substr(model_data$group, 1, 1) == i, ]

  xmax <- ifelse(i == 0, 50000, 220000)
  ymax <- ifelse(i == 0, 8000, 10000)
  ymax2 <- ifelse(i == 0, 50, 100)
  title_addition <- ifelse(i == 0, "", " (Multidirectional)")
  title <- paste("Embedding Model", title_addition, sep='')
  file_addition <- ifelse(i == 0, "-simple", "")
  if (i == 0) {
    model_baseline <- baseline_left
    model_perfect <- perfect_left
  } else {
    model_baseline <- baseline_difficult
    model_perfect <- perfect_directions_data[perfect_directions_data$group == -4,]
    model_perfect$metrics.cumulative_regret_total <- model_perfect$metrics.cumulative_regret_total / 5.0 # Account for sampling frequency
  }
  model_perfect$group <- paste(i, '-2');
  model_baseline$group <- paste(i, model_baseline$group)
  model_difficulty_data <- rbind(model_difficulty_data, model_baseline, model_perfect)
  
  model_labels <- setNames(
    c("Baseline", "Perfect", "Word2Vec", "Word2Vec + Pos", "InferSent", "SBERT", "SimCSE"),
    c(paste(i, -1), paste(i, -2), paste(i, 0), paste(i, 1), paste(i, 2), paste(i, 3), paste(i, 4))
  )
  
  # Plot models
  plot_regret(model_difficulty_data, model_labels, title, 6500, xmax, 0, ymax)
  save_plot(paste('embedding-model', file_addition, sep=''))
  
  plot_regret(model_difficulty_data, model_labels, title, 6500, xmax, 0, ymax2, FALSE)
  save_plot(paste('embedding-model', file_addition, '-regret', sep=''))
}

for (model_group in -1:5) {
  model_regrets <- model_data[model_data$group == paste(0, model_group) & model_data$z.env_steps == 50000, "metrics.cumulative_regret_total"]
  print(paste(model_group, ' $', round(mean(model_regrets), 0), ' \\pm ', round(sd(model_regrets), 0), '$', sep=''))
}

###################
# EMBEDDING WORDS #
###################
word_groups <- c(
  "words_word2vec" = 0,
  "words_word2vec_pos" = 1,
  "words_infersent" = 2,
  "words_sbert" = 3,
  "words_simcse" = 4
)
word_group <- group_by_property(word_groups, "env$task_file")
word_filepath = paste(input_folder, 'embedding-model-words\\embedding-model-words', sep='')
word_partial_data <- load_data(word_filepath, no_filter, word_group)
word_data <- rbind(word_partial_data, baseline_left)

word_labels <- c(
  "-1" = "Baseline", 
  "0" = "Word2Vec",
  "1" = "Word2Vec + Pos",
  "2" = "InferSent",
  "3" = "SBERT",
  "4" = "SimCSE"
)

plot_regret(word_data, word_labels, "Embedding Model (Words)", ymax=7000)
save_plot('embedding-words')

plot_regret(word_data, word_labels, "Embedding Model (Words)", ymax=50, cumulative=FALSE)
save_plot('embedding-words-regret')


########################
# INFORMATION DILUTION #
########################

# Object type, words, and sentences
dilution_filter <- function(yaml) {
  task_file <- yaml$env$task_file
  return(str_detect(task_file, "simcse"))
}
dilution_groups <- c(
  "sentences" = 0,
  "words" = 1,
  "object_type" = 2
)
dilution_group <- group_by_property(dilution_groups, "env$task_file")
dilution_filepath <- paste(input_folder, "embedding-type\\embedding-type-fifty-logs", sep='')
dilution_partial_data <- load_data(dilution_filepath, dilution_filter, dilution_group)

# Perfect embeddings
perfect_group <- function(yaml, data) {
  return(3);
}
perfect_filepath <- paste(d_input_folder, "embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-1\\rnn-0", sep='')
perfect_data <- load_data(perfect_filepath, no_filter, perfect_group)

# Pretrained model on sentences
pretrain_dilution_group <- function(yaml, csv) {
  return(4);
}
pretrain_dilution_filepath <- paste(input_folder, 'pretrain\\pretrain-logs\\logs\\meta\\obs-1\\rnn-0\\updates-0.05\\task-sentences_simcse\\selection-even', sep='')
pretrain_dilution_data <- load_data(pretrain_dilution_filepath, no_filter, pretrain_dilution_group)

dilution_no_baseline_data <- rbind(perfect_data, dilution_partial_data, pretrain_dilution_data)
dilution_data <- rbind(dilution_no_baseline_data, baseline_left)

dilution_labels <- c(
  "-1" = "Baseline",
  "0" = "Sentences",
  "1" = "Objects",
  "2" = "Object Types",
  "3" = "Perfect",
  "4" = "Pretrain"
)

make_boxplot(dilution_no_baseline_data, dilution_labels, "Information Dilution", skip_colors = 1, x_label = "Embeddeding Type")
save_plot('information-dilution-boxplot')

plot_regret(dilution_data, dilution_labels, "Information Dilution", 6500, 50000, 0, 5000)
save_plot('information-dilution')

plot_regret(dilution_data, dilution_labels, "Information Dilution", 6500, 50000, 0, 50, FALSE)
save_plot('information-dilution-regret')

# Outlier inspection
obj_seed_filter <- function(yaml) {
  task_file <- yaml$env$task_file
  return(str_detect(task_file, "object_type_simcse"))
}
objecttype_outlier_data <- load_outlier_data(objecttype_filepath, obj_seed_filter)
outlier_check(objecttype_outlier_data, 42, 89)

pretrain_outlier_data <- load_outlier_data(pretrain_dilution_filepath, no_filter)
outlier_check(pretrain_outlier_data, 42, 44)
outlier_check(pretrain_outlier_data, 47, 47)

##################
# GENERALIZATION #
##################
gen_filter <- function(yaml) {
  return(yaml$env$task_selection != "random")
}
# Load 30% and 80% results
gen_groups <- c(
  "random-word" = 0,
  "random-within-word" = 2
)
gen_group <- function(yaml, csv) {
  selection <- yaml$env$task_selection
  if (!(selection %in% names(gen_groups))) {
    stop(paste("Could not find group for '", paste(selection, "'", sep=""), sep=""))
  }
  level <- 0
  if (yaml$env$train_test_split == 0.3) {
    level <- 3
  }
  return(level + gen_groups[selection])
}
gen_filepath <- paste(input_folder, 'gen-tests\\generalization-logs', sep='')
gen_30p_80p_data <- load_data(gen_filepath, gen_filter, gen_group)

# Load 100% results
gen_100p_group <- function(yaml, csv) {
  return(6)
}
gen_100p_filepath <- paste(input_folder, 'embedding-type\\embedding-type-fifty-logs\\logs\\meta\\obs-1\\rnn-0\\updates-0.05\\task-sentences_simcse', sep='')
gen_100p_data <- load_data(gen_100p_filepath, no_filter, gen_100p_group)

# Load Pretrain results
pretrain_groups <- function(yaml, csv) {
  if(yaml$env$train_test_split == 0.3) {
    return(4);
  }
  if(yaml$env$train_test_split == 0.8) {
    return(1);
  }
  stop(paste('Train test split not recognized: \'', yaml$env$train_test_split, '\''))
}
pretrain_filepath <- paste(input_folder, "pretrain\\pretrain-logs\\logs\\meta\\obs-1\\rnn-0\\updates-0.05\\task-sentences_simcse\\selection-random-word", sep='')
pretrain_partial_data <- load_data(pretrain_filepath, no_filter, pretrain_groups)

gen_data <- rbind(gen_30p_80p_data, pretrain_partial_data, gen_100p_data, baseline_left)
#gen_data <- rbind(gen_30p_80p_data, baseline_left)

gen_80p_data <- gen_data[gen_data$group < 3 | gen_data$group == 6 | gen_data$group == -1, ]
gen_30p_data <- gen_data[gen_data$group >= 3 | gen_data$group == 6 | gen_data$group == -1, ]

gen_labels <- c(
  "-1" = "Baseline",
  "0" = "Unknown Objects",
  "1" = "Unknown Objects (Pre-trained)",
  "2" = "Unknown Formulations",
  "3" = "Unknown Objects",
  "4" = "Unknown Objects (Pre-trained)",
  "5" = "Unknown Formulations",
  "6" = "Seen All Sentences "
)
plot_regret(gen_80p_data, gen_labels, "Generalization, 20% Unseen", 6500, 100000, 0, 8500)
save_plot('generalization-20p')

plot_regret(gen_80p_data, gen_labels, "Generalization, 20% Unseen", 6500, 100000, 0, 50, FALSE)
save_plot('generalization-20p-regret')

plot_regret(gen_30p_data, gen_labels, "Generalization, 70% Unseen", 6500, 100000, 0, 8500)
save_plot('generalization-70p')

plot_regret(gen_30p_data, gen_labels, "Generalization, 70% Unseen", 6500, 100000, 0, 50, FALSE)
save_plot('generalization-70p-regret')


# Outlier inspection
gen_seed_filter <- function(yaml) {
  return(yaml$env$task_selection == "random" && yaml$env$train_test_split == 0.3);
}
gen_outlier_data <- load_outlier_data(gen_filepath, gen_seed_filter)
outlier_check(gen_outlier_data, 42, 89)
outlier_check(gen_outlier_data, 46, 46)

##############
# CLASSIFIER #
##############

# Classifier embedding type
classifier_type_groups <- c(
  "embeddings/one_direction/sentences_simcse.dill" = 0,
  "embeddings/one_direction/sentences_word2vec.dill" = 1,
  "embeddings/one_direction/words_simcse.dill" = 2,
  "embeddings/one_direction/words_word2vec.dill" = 3,
  "embeddings/one_direction/perfect.dill" = 4
)
classifier_type_group <- function(yaml, csv) {
  return(classifier_type_groups[csv$file])
}

classifier_type_filepath <- paste(input_folder, "classifier\\embedding-type", sep='')
classifier_type_data <- load_data(classifier_type_filepath, no_filter, classifier_type_group, FALSE)
classifier_type_labels <- c(
  "0" = "Sentences SimCSE",
  "1" = "Sentences Word2Vec",
  "2" = "Words SimCSE",
  "3" = "Words Word2Vec",
  "4" = "Perfect"
)
classifier_type_data$z.env_steps <- classifier_type_data$z.env_steps + 1
make_plot(classifier_type_data, classifier_type_labels, "Classification by Policy Network", 0, 20, 0.5, 1.03, 'metrics.eval_accuracy', 'Test Accuracy', 'Epochs',
          c(0.98, 0.02), c("right", "bottom"), "right")
save_plot('classifier-type')


# Classifier generalization
classifier_networks <- c(
  "small" = 0, # No hidden layers
  "policy" = 1, # 2 hidden layers
  "big" = 2 # 7 hidden layers
)
classifier_task_selections <- c(
  "random" = 0,
  "random-word" = 1,
  "random-within-word" = 2
)
classifier_gen_group <- function(yaml, csv) {
  return(paste(classifier_networks[yaml$network], classifier_task_selections[csv$task_selection], csv$split));
}
classifier_gen_filepath <- paste(input_folder, "classifier\\generalization", sep = '')
classifier_gen_data <- load_data(classifier_gen_filepath, no_filter, classifier_gen_group, FALSE)
classifier_gen_data$z.env_steps <- classifier_gen_data$z.env_steps + 1
classifier_gen_labels <- c(
  "0.1" = "10%",
  "0.3" = "30%",
  "0.5" = "50%",
  "0.8" = "80%",
  "1" = "100%"
)
classifier_gen_titles <- c(
  "random" = "Uncontrolled",
  "random-word" = "Unknown Objects",
  "random-within-word" = "Unknown Formulations"
)
classifier_network_titles <- c(
  "small" = "0 Hidden Layers",
  "policy" = "2 Hidden Layers",
  "big" = "7 Hidden Layers"
)
for (network in names(classifier_networks)) {
  for (task_selection in names(classifier_task_selections)) {
    print(paste("Network: '", network, "', Task selection: '", task_selection, "'", sep=''))
    classifier_group_data <- classifier_gen_data[
      substr(classifier_gen_data$group, 1, 1) == classifier_networks[network] &
      substr(classifier_gen_data$group, 3, 3) == classifier_task_selections[task_selection], 
    ]
    classifier_group_data$group <- substring(classifier_group_data$group, 5)
    classifier_group_data <- filter(classifier_group_data, group == "0.1" | group == "0.3" | group == "0.5" | group == "0.8" | group == "1")
    
    classifier_group_title <- paste("Classification with", classifier_network_titles[network], "on", classifier_gen_titles[task_selection])
    make_plot(classifier_group_data, classifier_gen_labels, classifier_group_title, 0, 40, 0.5, 1.03, 'metrics.eval_accuracy', 'Validation Accuracy', 'Epochs',
              c(0.98, 0.02), c("right", "bottom"), "right")
    save_plot(paste('classifier-gen-', network, '-', task_selection, sep=''))
  }
}

###############
# NUM UPDATES #
###############
num_updates_group <- function(yaml, csv) {
  return(yaml$train$num_updates_per_iter)
}
num_updates_filepath <- paste(input_folder, "num-updates\\num-updates-logs", sep = '')
num_updates_data <- load_data(num_updates_filepath, no_filter, num_updates_group)
num_updates_labels <- c(
  "0.025" = "0.025",
  "0.05" = "0.05",
  "0.075" = "0.075",
  "0.1" = "0.1",
  "0.15" = "0.15",
  "0.125" = "0.125",
  "0.175" = "0.175",
  "0.2" = "0.2"
)
plot_regret(num_updates_data, num_updates_labels, "Number of Updates per Environment Step", 6500, 50000, 0, 6500)
save_plot('num-updates')

plot_regret(num_updates_data, num_updates_labels, "Number of Updates per Environment Step", 6500, 50000, 0, 100, FALSE)
save_plot('num-updates-regret')

##############
# DIRECTIONS #
##############
directions_groups <- c(
  "left_directions" = 3,
  "leftright_directions" = 2,
  "all_directions_negation" = 0,
  "all_directions" = 1 # This one must come last
)
directions_groups <- c(
  "left" = 3,
  "leftright" = 2,
  "all_directions_negation" = 0,
  "all_directions" = 1 # This one must come last
)
directions_group <- group_by_property(directions_groups, "env$task_file")
directions_filepath <- paste(input_folder, "directions\\directions-log", sep = '')
directions_data <- load_data(directions_filepath, no_filter, directions_group)

get_gaps <- function(perfect_group, directions_group, source_directions=directions_data) {
  perfect_temp_data <- perfect_directions_data[perfect_directions_data$group == perfect_group,] 
  temp_data <- source_directions[source_directions$group == directions_group, ]
  cum_gap <- get_perfect_gap(temp_data, 'metrics.cumulative_regret_total', perfect_temp_data)
  gap <- get_perfect_gap(temp_data, 'metrics.regret', perfect_temp_data)
  return(list(cum_gap, gap))
}

c(directions_left_cum_gap, directions_left_gap) %<-% get_gaps(-2, 3)
c(directions_leftright_cum_gap, directions_leftright_gap) %<-% get_gaps(-3, 2)
c(directions_all_cum_gap, directions_all_gap) %<-% get_gaps(-4, 1)
c(directions_all_negation_cum_gap, directions_all_negation_gap) %<-% get_gaps(-4, 0)
#c(directions_baseline_cum_gap, directions_baseline_gap) %<-% get_gaps(-4, -1, baseline_difficult)

directions_cum_gap <- rbind(directions_left_cum_gap, directions_leftright_cum_gap, directions_all_cum_gap, directions_all_negation_cum_gap)
directions_gap <- rbind(directions_left_gap, directions_leftright_gap, directions_all_gap, directions_all_negation_gap)
directions_labels <- c(
  "0" = "All & Negation",
  "1" = "All",
  "2" = "Left & Right",
  "3" = "Left"
)
draw_plot(directions_cum_gap, directions_labels, "Information Compression", xmax=200000, ymax=15000, target_label="Cumulative Regret Gap")
save_plot('directions')

draw_plot(directions_gap, directions_labels, "Information Compression", xmax=200000, ymax=50, target_label="Regret Gap", position=c(0.98, 1), justification=c("right", "top"), box_just="right")
save_plot('directions-regret')


#########################
# DIRECTIONS EVALUATION #
#########################
directions_eval_groups <- c(
  "left_allstraight.dill" = 0,
  "leftright_allstraight.dill" = 1,
  "left_longstraight.dill" = 2,
  "leftright_longstraight.dill" = 3,
  "left_longhook.dill" = 4,
  "leftright_longhook.dill" = 5
)
directions_eval_group = group_by_property(directions_eval_groups, "env$task_file")
directions_eval_filepath <- paste(d_input_folder, "directions\\directions-eval-log", sep='')
directions_eval_data <- load_data(directions_eval_filepath, no_filter, directions_eval_group)

directions_eval_labels <- c(
  "0" = "No Bends, Left",
  "1" = "No Bends, Left & Right",
  "2" = "1 Bend, Left",
  "3" = "1 Bend, Left & Right",
  "4" = "2 Bends, Left",
  "5" = "2 Bends, Left & Right"
)
plot_regret(directions_eval_data, directions_eval_labels, "Number of Bends for Directions", 6500, 190000, 0, 50000)
save_plot('directions-eval')

plot_regret(directions_eval_data, directions_eval_labels, "Number of Bends for Directions", 6500, 190000, 0, 100, FALSE)
save_plot('directions-eval-regret')

###############
# UNCERTAINTY #
###############
uncertainty_group <- function(yaml, csv) {
  return(yaml$policy$uncertainty$scale);
}
uncertainty_filepath <- paste(d_input_folder, "uncertainty\\uncertainty-logs", sep='');
uncertainty_data <- load_data(uncertainty_filepath, no_filter, uncertainty_group)
uncertainty_labels <- c(
  "0" = "0",
  "0.001" = "0.001",
  "0.002" = "0.002",
  "0.005" = "0.005",
  "0.01" = "0.01",
  "0.2" = "0.2",
  "0.5" = "0.5",
  "1" = "1",
  "2" = "2",
  "5" = "5",
  "10" = "10",
  "20" = "20",
  "50" = "50",
  "100" = "100",
  "200" = "200",
  "500" = "500",
  "1000" = "1000"
)

low_uncertainty_data <- uncertainty_data[uncertainty_data$group <= 2, ]
high_uncertainty_data <- uncertainty_data[uncertainty_data$group > 2, ]

plot_regret(low_uncertainty_data, uncertainty_labels, "Uncertainty", 6500, 60000, 0, 10000)
save_plot('uncertainty-low')

plot_regret(low_uncertainty_data, uncertainty_labels, "Uncertainty", 6500, 60000, 0, 100, FALSE)
save_plot('uncertainty-low-regret')

plot_regret(high_uncertainty_data, uncertainty_labels, "Uncertainty", 6500, 60000, 0, 55000)
save_plot('uncertainty-high')

plot_regret(high_uncertainty_data, uncertainty_labels, "Uncertainty", 6500, 60000, 0, 200, FALSE)
save_plot('uncertainty-high-regret')

##############
# ACTIVATION #
##############
activation_groups <- c(
  "directly" = 0,
  "no-grad" = 1,
  "grad-relu6" = 8,
  "grad-relu" = 2,
  "grad-leaky" = 3,
  "grad-swish" = 4,
  "grad-elu" = 5,
  "grad-selu" = 6,
  "grad-gelu" = 7,
  "grad" = 2
)
activation_group <- group_by_property(activation_groups, "policy$embedding_grad")
activation_filepath <- paste(d_input_folder, 'activations\\activations-logs', sep='')
activation_partial_data <- load_data(activation_filepath, no_filter, activation_group)

grad_filepath <- paste(input_folder, "embedding-type\\embedding-type-fifty-logs\\logs\\meta\\obs-1\\rnn-0\\updates-0.05\\task-sentences_simcse\\selection-even", sep='')
grad_data <- load_data(grad_filepath, no_filter, activation_group)
activation_data <- rbind(activation_partial_data, grad_data)

activation_labels <- c(
  "2" = "ReLU",
  "0" = "Directly",
  "1" = "Identity",
  "3" = "Leaky ReLU",
  "4" = "Swish",
  "5" = "eLU",
  "6" = "SeLU",
  "7" = "GeLU",
  "8" = "ReLU6"
)

plot_regret(activation_data, activation_labels, "Activation Function", 6500, 50000, 0, 3500)
save_plot('activation')

plot_regret(activation_data, activation_labels, "Activation Function", 6500, 50000, 0, 50, FALSE)
save_plot('activation-regret')

make_boxplot(activation_data, activation_labels, "Activation Function", x_label = "Activation Function")
save_plot('activation-boxplot')

# Outlier detection
activations_filter <- function(yaml) {
  return(yaml$policy$embedding_grad == "grad-elu")
}
act_outlier_data <- load_outlier_data(activation_filepath, activations_filter)
outlier_check(act_outlier_data, 48, 50, 50000)
outlier_check(act_outlier_data, 53, 53, 50000)


##################
# PERFECT ONEHOT #
##################
perfect_onehot_group <- function(yaml, csv) {
  return(-1);
}
perfect_onehot_filepath <- paste(input_folder, 'perfect\\perfect-onehot', sep='')
perfect_onehot_data <- load_data(perfect_onehot_filepath, no_filter, perfect_onehot_group)

plot_regret(perfect_onehot_data, c(), "Perfect OneHot", ymax=4000)
save_plot('perfect-onehot')

#############
# TIME COST #
#############
time_directly_filter <- function(yaml) {
  return(yaml$policy$embedding_grad == 'directly');
}
time_directly_group <- function(yaml, csv) {
  return(1);
}
time_directly_filepath <- paste(d_input_folder, 'activations\\activations-logs', sep='');
time_directly_data <- load_data(time_directly_filepath, time_directly_filter, time_directly_group, only_necessary_columns = FALSE)
time_directly_data$group <- 0

time_obs_data <- load_data(perfect_left_filepath, no_filter, perfect_group, only_necessary_columns = FALSE)
time_obs_data$group <- 1

time_lstm_filepath <- paste(d_input_folder, "embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-0\\rnn-1", sep='')
time_lstm_data <- load_data(time_lstm_filepath, no_filter, time_directly_group, only_necessary_columns = FALSE)
time_lstm_data$group <- 2

time_data <- rbind(time_directly_data, time_lstm_data, time_obs_data)
time_data$time <- time_data$z.time_cost / 3600
time_labels <- c(
  "2" = "LSTM Hidden State Init, 86k parameters",
  "1" = "Obs Concat via Linear Layer, 103k parameters",
  "0" = "Obs Concat Directly, 611k parameters"
)

make_plot(time_data, time_labels, "Training Time of LSTM Sizes", 6500, 50000, 0, 24, "time", "Time Cost (hr)", "Environment Steps", c(0.02, 1), c("left", "top"), "left")
save_plot('lstm-size-timecost')

#################
# VALID ACTIONS #
#################
valid_actions_groups <- c(
  "_all_" = 0,
  "_leftright_" = 1,
  "_left_" = 2
)
valid_actions_group <- group_by_property(valid_actions_groups, "env$task_file")
valid_actions_filepath <- paste(input_folder, 'valid-actions\\valid-actions-directions', sep='')
valid_actions_partial_data <- load_data(valid_actions_filepath, no_filter, valid_actions_group)
valid_actions_data <- rbind(valid_actions_partial_data, perfect_directions_data)

valid_actions_labels <- c(
  "0" = "All Valid",
  "1" = "Left&Right Valid",
  "2" = "Left Valid",
  "-4" = "All",
  "-3" = "Left&Right",
  "-2" = "Left"
)
valid_actions_title <- "Limit Actor to Choosing Valid Actions"
plot_regret(valid_actions_data, valid_actions_labels, valid_actions_title, xmax=200000, ymax=60000)
save_plot('valid-actions')

plot_regret(valid_actions_data, valid_actions_labels, valid_actions_title, xmax=200000, ymax=200, cumulative=FALSE)
save_plot('valid-actions-regret')

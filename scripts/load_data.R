group_by_property <- function(groups, column) {
  handler <- function(yaml, csv) {
    
    # Extract the property from the yaml list
    property <- yaml;
    for (column_part in strsplit(column, "\\$")) { 
      property <- property[[column_part]]
    }
    
    # Find the property in the groups
    name <- NULL
    for (group_name in names(groups)) {
      if (str_detect(property, group_name)) {
        print(group_name)
        return(groups[group_name])
      }
    }
    stop(paste("Cannot find group for property '", property, "'", sep=''));
  }
  return(handler);
}

# Function to read and process CSV and YAML data
read_process_data <- function(csv_file, yaml_file, filter_callback, group_callback) {
  yaml_data <- yaml::yaml.load_file(yaml_file)
  if (!filter_callback(yaml_data)) {
    return(NULL);
  }
  data <- read.csv(csv_file)
  data$group <- group_callback(yaml_data, data);
  return(data)
}

# Function to traverse directories and find CSV and YAML pairs
find_csv_yaml_pairs <- function(directory) {
  files <- list.files(directory, full.names = TRUE)
  subdirs <- list.dirs(directory, full.names = TRUE, recursive = FALSE)
  
  csv_file <- NULL
  yaml_file <- NULL
  
  for (file in files) {
    if (tools::file_ext(file) == "csv" && str_detect(file, "progress-filled.")) {
      if (!is.null(csv_file)) {
        stop(paste("Found 2 csv files '", file, "' and '", csv_file, "'", sep=''))
      }  
      csv_file <- file
    } else if (tools::file_ext(file) %in% c("yaml", "yml")) {
      if (!is.null(yaml_file)) {
        stop(paste("Found 2 yaml files '", file, "' and '", yaml_file, "'", sep=''))
      }
      yaml_file <- file
    }
  }
  if (is.null(csv_file) != is.null(yaml_file)) {
    stop(paste("Found a non-matching set! ", paste(csv_file, yaml_file)))
  }
  
  # Match CSV and YAML files based on 'embedding_obs_init' and 'embedding_rnn_init'
  matched_pairs <- NULL
  if (!is.null(csv_file)) {
    matched_pairs <- c(csv_file, yaml_file)
  }
  
  # Recursively search subdirectories
  for (subdir in subdirs) {
    new_pair <- find_csv_yaml_pairs(subdir)
    if (!is.null(new_pair)) {
      if (!is.null(matched_pairs)) {
        matched_pairs <- c(matched_pairs, new_pair)
      } else {
        matched_pairs <- new_pair
      }
    }
  }
  
  matched_pairs
}

load_data <- function(directory, filter_callback, group_callback, only_necessary_columns =TRUE) {
  # Find CSV and YAML pairs
  csv_yaml_pairs <- find_csv_yaml_pairs(directory)
  
  # Create an empty data frame to store all data
  all_data <- data.frame()
  
  # Read and process data for each pair
  for (index in seq(1, length(csv_yaml_pairs), 2)) {
    data <- read_process_data(
      csv_yaml_pairs[index], 
      csv_yaml_pairs[index + 1],
      filter_callback,
      group_callback
    )
    if (is.null(data)) {
      next
    }
    if (only_necessary_columns) {
      columns <- c('metrics.regret', 'metrics.cumulative_regret_total', 'z.env_steps', 'group')
      data <- data[, names(data) %in% columns]
    }
    all_data <- rbind(all_data, data)
  }
  
  all_data
}
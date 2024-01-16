plot_regret <- function(data, labels, title, xmin=6500, xmax=50000, ymin=0, ymax=30000, cumulative=TRUE) {
  if (cumulative) {
    make_plot(data, labels, title, xmin, xmax, ymin, ymax, "metrics.cumulative_regret_total", "Cumulative Regret", "Environment Steps", c(0.02, 1), c("left", "top"), "left")
  } else {
    make_plot(data, labels, title, xmin, xmax, ymin, ymax, 'metrics.regret', 'Episodic Regret', "Environment Steps", c(0.98, 1), c("right", "top"), "right")
  }
}


make_plot <- function(data, labels, title, xmin=6500, xmax=50000, ymin=0, ymax=30000, target_column = "metrics.cumulative_regret_total", target_label = "Cumulative Regret", x_label = "Environment Steps", 
                      position = c(0.05, 1), justification = c("left", "top"), box_just = "left", skip_colors = NULL, perfect_data = NULL) {
  ggthemr('fresh')
  grouped_data <- get_perfect_gap(data, target_column, perfect_data)
  
  # Create a line graph with error bars using ggplot2
  draw_plot(grouped_data, labels, title, xmin, xmax, ymin, ymax, target_label, x_label, position, justification, box_just, skip_colors)
}

group_data <- function(data, target_column) {
  grouped_data <- data %>%
    group_by(group, z.env_steps) %>%
    summarize(
      mean_value = mean(.data[[target_column]]),
      std_error = sd(.data[[target_column]]) / sqrt(n())  # Calculate standard error
    )
  return(grouped_data)
}

get_perfect_gap <- function(data, target_column, perfect_data = NULL) {
  grouped_data <- group_data(data, target_column)
  
  if (!is.null(perfect_data)) {
    print("Subtracting perfect data...")
    # Make both datasets the same size
    grouped_perfect_data <- group_data(perfect_data, target_column)
    
    # Trim both to be the same size
    grouped_data <- grouped_data[grouped_data$z.env_steps <= max(grouped_perfect_data$z.env_steps), ]
    grouped_perfect_data <- grouped_perfect_data[grouped_perfect_data$z.env_steps <= max(grouped_data$z.env_steps), ]
    
    for (group in unique(data$group)) {
      grouped_data[grouped_data$group == group, "mean_value"] <- grouped_data[grouped_data$group == group, "mean_value"] - grouped_perfect_data[, "mean_value"]
    }
  }
  
  return(grouped_data);
}

make_boxplot <- function(data, labels, title, x_pos = 50000, target_column = "metrics.cumulative_regret_total", target_label = "Cumulative Regret", x_column = 'z.env_steps', x_label = "Environment Steps", skip_colors = 0) {
  filtered_data <- data[data[x_column] == x_pos,]
  filtered_data$group <- factor(filtered_data$group)
  
  num_groups <- length(unique(filtered_data$group))
  pal_fill_all <- scales::brewer_pal(palette="Set1")(num_groups + skip_colors)
  pal_fill <- pal_fill_all[(skip_colors + 1):(num_groups + 1)]
  
  ggplot(filtered_data, aes(x = group, y = .data[[target_column]], group = group, fill = group)) +
    geom_boxplot() +
    labs(title = title,
         labels = labels,
         x = x_label,
         y = target_label) +
    scale_x_discrete(labels = labels) +
    scale_fill_manual(values = pal_fill) + 
    theme(legend.position = "none")
}

draw_plot <- function(grouped_data, labels, title, xmin=6500, xmax=50000, ymin=0, ymax=30000, target_label = "Cumulative Regret", x_label = "Environment Steps", 
                      position = c(0.05, 1), justification = c("left", "top"), box_just = "left", skip_colors = NULL) {
  pal_fill <- scales::brewer_pal(palette="Set1")
  grouped_data$group <- factor(grouped_data$group)
  ggplot(grouped_data, aes(x = z.env_steps, y = mean_value, color = group)) +
    geom_line() +
    geom_ribbon(aes(ymin = mean_value - 2.01174 * std_error,
                    ymax = mean_value + 2.01174 * std_error,
                    fill=group), alpha = 0.2, linetype = 0) +
    scale_colour_brewer(palette="Set1",labels = labels, name = NULL) +
    scale_fill_brewer(palette="Set1",labels = labels, name = NULL) +
    labs(
      x = x_label,
      y = target_label,
      title = title,
      color = "group"
    ) +
    coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) +
    theme(
      legend.background = element_rect(
        fill = "white", 
        linewidth = 4, 
        colour = "white"
      ),
      legend.position = position,
      legend.justification = justification,
      legend.box.just = box_just,
      axis.ticks = element_line(colour = "grey70", linewidth = 0.2),
      panel.grid.major = element_line(colour = "grey70", linewidth = 0.2),
      panel.grid.minor = element_blank()
    )
}
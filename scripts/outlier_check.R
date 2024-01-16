library(RColorBrewer)

load_outlier_data <- function(filepath, filter_callback) {
  seed_group <- function(yaml, csv) {
    return(yaml$seed);
  }
  return(load_data(filepath, filter_callback, seed_group))
}
outlier_check <- function(outlier_data, seedmin = 42, seedmax = 89, xmax = 100000) {
  seed_labels <- list()
  for (i in 42:89) {
    seed_labels[toString(i)] <- toString(i)
  }
  
  seed_data <- filter(outlier_data, group >= seedmin & group <= seedmax)
  outlier_plotter(seed_data, seed_labels, "Seeds", 6500, xmax, 0, 100, 'metrics.regret', 'Episode Regret')
}

outlier_plotter <- function(data, labels, title, xmin=6500, xmax=35000, ymin=0, ymax=30000, target_column = "metrics.cumulative_regret_total", target_label = "Cumulative Regret", x_label = "Environment Steps", legpos_x = 0, legpos_y=1) {
  grouped_data <- data %>%
    group_by(group, z.env_steps) %>%
    summarize(
      mean_value = mean(.data[[target_column]]),
    )
  
  grouped_data$group <- factor(grouped_data$group)
  # Create a line graph with error bars using ggplot2
  colors <- c(
    brewer.pal(name="Set1", n = 9),
    brewer.pal(name="Set2", n = 8), # 17
    brewer.pal(name="Set3", n = 12), # 19
    brewer.pal(name="Paired", n = 12), # 31
    brewer.pal(name="Pastel1", n = 9), # 40
    brewer.pal(name="Pastel2", n = 8) # 48
  )
  ggplot(grouped_data, aes(x = z.env_steps, y = mean_value)) +
    geom_line() +
    labs(
      x = x_label,
      y = target_label,
      title = title,
    ) +
    coord_cartesian(ylim = c(ymin, ymax), xlim = c(xmin, xmax)) +
    theme(
      legend.background = element_rect(
        fill = "white", 
        linewidth = 4, 
        colour = "white"
      ),
      legend.justification = c(-0.05, 1),
      legend.position = "none",
      axis.ticks = element_line(colour = "grey70", linewidth = 0.2),
      panel.grid.major = element_line(colour = "grey70", linewidth = 0.2),
      panel.grid.minor = element_blank()
    ) +
    scale_color_manual(colors)
}
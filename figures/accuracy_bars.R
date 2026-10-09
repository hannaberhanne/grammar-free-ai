suppressPackageStartupMessages(library(ggplot2))

chart_data <- data.frame(
  model = factor(c("PDA", "FSA", "CFG"), levels = c("PDA", "FSA", "CFG")),
  accuracy = c(1.00, 7 / 12, 0.80),
  label = c("100%", sprintf("%.1f%%", 100 * 7 / 12), "80%"),
  fill = c("#2E8B57", "#D95F02", "#4C78A8")
)

plot_accuracy <- ggplot(chart_data, aes(x = model, y = accuracy, fill = model)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_text(
    aes(label = label),
    vjust = -0.55,
    size = 4.4,
    fontface = "bold",
    color = "#1F1F1F"
  ) +
  scale_fill_manual(values = setNames(chart_data$fill, chart_data$model)) +
  scale_y_continuous(
    limits = c(0, 1.08),
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = function(x) paste0(round(x * 100), "%"),
    expand = c(0, 0)
  ) +
  labs(
    title = "Formal Model Accuracy",
    x = NULL,
    y = "Accuracy"
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    axis.title.y = element_text(face = "bold", color = "#2B2B2B", margin = margin(r = 10)),
    axis.text.x = element_text(face = "bold", color = "#2B2B2B"),
    axis.text.y = element_text(color = "#4A4A4A"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "#D9D9D9", linewidth = 0.5),
    axis.line.x = element_line(color = "#2B2B2B", linewidth = 0.4),
    axis.line.y = element_line(color = "#2B2B2B", linewidth = 0.4),
    plot.margin = margin(18, 16, 12, 12),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

ggsave(
  filename = "accuracy_bars.png",
  plot = plot_accuracy,
  width = 4.8,
  height = 7.0,
  dpi = 200,
  bg = "transparent"
)

ggsave(
  filename = "accuracy_bars.pdf",
  plot = plot_accuracy,
  width = 4.8,
  height = 7.0,
  bg = "transparent"
)

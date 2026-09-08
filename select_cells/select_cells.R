library(plotly)
library(shiny)
library(rhdf5)
library(RColorBrewer)
library(stringr)
library(pals)
color_scheme = brewer.set1(10)
####select cell to perturb###
data = H5Fopen('../datasets/GSM5808054_data.h5')
df_ = data.frame(x=data$pos[,2], y=-data$pos[,1], tissue=data$tissue, perturbation=data$perturbation, index = seq_along(data$tissue), size=10, selected = FALSE)

######################################################

ui <- fluidPage(
  titlePanel("Lasso Select Cells"),
  
  plotlyOutput("dotplot", width = "1000px", height = "800px"),
  
  verbatimTextOutput("selected_indices"),  # Display selected indices
  actionButton("save_indices", "Save Selected Indices"),  # Button to save selection
  verbatimTextOutput("saved_indices"),  # Display saved indices
  
  # Download button
  downloadButton("download_indices", "Download Selected Indices")
)

server <- function(input, output, session) {
  df_reactive <- reactiveVal(df_)  # Store the dataset
  
  # Reactive values to store selected and saved indices
  selected_indices <- reactiveVal(c())  
  saved_indices <- reactiveVal(c())  
  
  # Render the interactive plot
  output$dotplot <- renderPlotly({
    df <- df_reactive()
    plot_ly(df, x = ~x, y = ~y, type = 'scatter', mode = 'markers',
            color = ~perturbation, colors = color_scheme, size = ~size) %>%
      layout(title = "Lasso or Click to Select Points") %>%
      config(modeBarButtonsToAdd = list('lasso2d', 'select2d')) %>%
      event_register("plotly_selected") %>%
      event_register("plotly_click")  # <- Register click event
  })
  
  # Handle lasso/box selection
  observeEvent(event_data("plotly_selected"), {
    selected_data <- event_data("plotly_selected")
    if (!is.null(selected_data)) {
      df <- df_reactive()
      selected_indices_new <- df$index[match(paste(selected_data$x, selected_data$y), paste(df$x, df$y))]
      selected_indices(unique(selected_indices_new))  # replace selection
      df$size <- ifelse(df$index %in% selected_indices_new, 12, 10)
      df_reactive(df)
    }
  })
  
  # Handle single-click selection (adds to current selection)
  observeEvent(event_data("plotly_click"), {
    click_data <- event_data("plotly_click")
    if (!is.null(click_data)) {
      df <- df_reactive()
      clicked_index <- df$index[which.min((df$x - click_data$x)^2 + (df$y - click_data$y)^2)]
      
      current <- selected_indices()
      new_selection <- unique(c(current, clicked_index))  # add to existing selection
      
      selected_indices(new_selection)
      df$size <- ifelse(df$index %in% new_selection, 12, 10)
      df_reactive(df)
    }
  })
  
  output$selected_indices <- renderPrint({
    indices <- selected_indices()
    if (length(indices) == 0) "No points selected yet." else indices
  })
  
  observeEvent(input$save_indices, {
    saved_indices(selected_indices())  
  })
  
  output$saved_indices <- renderPrint({
    saved <- saved_indices()
    if (length(saved) == 0) "No indices saved yet." else saved
  })
  
  output$download_indices <- downloadHandler(
    filename = function() {
      paste0("selected_indices_", Sys.Date(), ".txt")
    },
    content = function(file) {
      writeLines(as.character(saved_indices()), file)  
    }
  )
}

output = shinyApp(ui, server)# Run the Shiny app

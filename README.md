
## Introduction

The ART-TREES Standard V2.01 mandates specific methodologies for calculating and reporting uncertainty estimates associated with emission factors and activity data within jurisdictional and nested REDD+ projects. To strengthen compliance, the ART-TREES project team produced the following report and capacity building resources.

## Example script {#sec-1.1}

#### Environment setup

```{r setup-2}
#| warning: false
#| message: false
#| error: false
#| include: true
#| echo: true
#| eval: false
easypackages::packages(
  "animation", "BIOMASS", "caret", "dataMaid", "DescTools", "dplyr",
  "extrafont", "FawR", "ForestToolsRS", "ggplot2", "htmltools",
  "janitor", "jsonlite", "lattice", "kableExtra", "kernlab",
  "knitr", "Mlmetrics", "olsrr", "plotly", "psych", "RColorBrewer",
  "rmarkdown", "readxl", "solarizeddox", "tibble", "tidymodels", "tidyverse",
  "tinytex", "tune", "useful", "webshot", "webshot2",
  prompt = F
)
```

### Monte Carlo of Emissions Factors

#### Import data

This section outlines the tools for importing and preparing forestry and biomass
data for analysis, a key step in building ART-TREES-compliant MRV systems. Using
the `allodb` package, we load a global allometry database and a dummy dataset
from the Smithsonian Institute ForestGEO project.

```{r dummy-import}
#| warning: false
#| message: false
#| error: false
#| echo: true
library("allodb") # https://docs.ropensci.org/allodb/
set.seed(333)
#data(ufc) # spuRs::vol.m3(dataset$dbh.cm, dataset$height.m, multiplier = 0.5)
data(scbi_stem1)
dataset = scbi_stem1
head(dataset) |> tibble::as_tibble()

psych::describe(dataset)
str(dataset)
```

##### Table 3: Smithsonian Institute GEOForest dataset from `allodb` package (n = 2287)

#### Probability density functions

Accurate selection of probability density functions (PDFs) is essential for
modeling uncertainties in carbon stocks and activity data. This section
describes methodologies for fitting PDFs to data, ensuring results are robust
and aligned with ART-TREES best practices.

-   Use of statistical tests for goodness-of-fit validation.

-   Integration of domain expertise to refine parameter selection.

```{r, fig.show='hold', out.width="33%"}
# add allometry database
data(equations)
data("equations_metadata")
show_cols   = c("equation_id", "equation_taxa", "equation_allometry")
eq_tab_acer = new_equations(subset_taxa = "Acer")
head(eq_tab_acer[, show_cols])

# Compute above ground biomass
dataset$agb = allodb::get_biomass(
  dbh     = dataset$dbh,
  genus   = dataset$genus,
  species = dataset$species,
  coords  = c(-78.2, 38.9)
)

# examine dbh ~ agb function
dbh_agb = lm(dbh ~ agb, data = dataset)
#olsrr::ols_test_breusch_pagan(lm(dbh_agb)) #<0.0000
#h = lattice::histogram(dbh ~ agb, data = dataset)
plot(
  x    = dataset$dbh,
  y    = dataset$agb,
  col  = factor(scbi_stem1$genus),
  xlab = "DBH (cm)",
  ylab = "AGB (kg)"
)

# examine univariate distributions
h1 = hist(dataset$dbh, breaks=10, col="red")
xfit<-seq(min(dataset$dbh),max(dataset$dbh),length=40)
yfit<-dnorm(xfit,mean=mean(dataset$dbh),sd=sd(dataset$dbh))
yfit <- yfit*diff(h1$mids[1:2])*length(dataset$dbh)
lines(xfit, yfit, col="blue", lwd=2)

h2 = hist(dataset$agb, breaks=10, col="red")
xfit<-seq(min(dataset$agb),max(dataset$agb),length=40)
yfit<-dnorm(xfit,mean=mean(dataset$agb),sd=sd(dataset$agb))
yfit <- yfit*diff(h2$mids[1:2])*length(dataset$agb)
lines(xfit, yfit, col="blue", lwd=2)
wilcox.test(dataset$dbh) # p<0.00001
wilcox.test(dataset$agb) # p<0.00001
```

#### Simulation Design

This section introduces the design of the Monte Carlo simulation regime,
including:
  
  -   Simulation parameters are defined to balance computational efficiency and
statistical robustness.

-   Cross-validation techniques are employed to evaluate model performance and
identify bias or variance.

The `LGOCV` acronym used in the `caret` package functions below stands for
"leave one group out cross validation". We must select the % of test data that
is set out from the build upon which the model will be repeatedly trained. Note,
the following code applies functions to full dataset without explicit
training-test split. **Questions remains on whether we require cross-validation
uncertainty estimate to review internal bias, and whether we would like to
develop Monte Carlo tools for spatial uncertainty used in Activity Data
analysis**. For your consideration, the consultant has previously developed
Monte Carlo tools for LULC applications, saved
[here](https://github.com/seamusrobertmurphy/02-lulc-classification)

```{r}
# Cross-validation split for bias detection
#samples     = caret::createDataPartition(dataset_tidy$volume, p = 0.80, list = FALSE)
#train_data  = dataset_tidy[samples, ]
#test_data   = dataset_tidy[-samples, ]

# Simulation pattern & regime
monte_carlo = trainControl(
  method    = "LGOCV",
  number    = 10,     # number of simulations
  p         = 0.8)     # percentage resampled


# Training model fit with all covariates (".") & the simulation
lm_monte_carlo = train(
  data      = dataset, 
  agb ~ ., 
  na.action = na.omit,
  trControl = monte_carlo)

lm_monte_carlo 
```

#### Visualize residuals

To enable access to these predictions, we need to instruct `caret` to retain the
resampled predictions by setting `savePredictions = "final"` in our
`trainControl()` function. It's important to be aware that if you’re working
with a large dataset or numerous resampling iterations, the resulting `train()`
object may grow significantly in size. This happens because `caret` must store a
record of every row, including both the observed values and predictions, for
each resampling iteration. By visualizing the results, we can offer insights
into the performance of our model on the resampled data.

```{r}
monte_carlo_viz = trainControl(
  method    = "LGOCV", 
  p         = 0.8,            
  number    = 1,  # just for saving previous results
  savePredictions = "final") 

lm_monte_carlo_viz = train(
  agb ~ ., 
  data      = dataset, 
  method    = "lm",
  na.action = na.omit,
  trControl = monte_carlo_viz)

head(lm_monte_carlo_viz$pred)  # residuals 

lm_monte_carlo_viz$pred |> 
  ggplot(aes(x=pred,y=obs)) +
    geom_point(shape=1) + 
    geom_abline(slope=1, colour='blue')  +
    coord_obs_pred()
```

### Activity Data Uncertainty

This section showcases use of Monte Carlo simulations in reporting uncertainty
of LULC classification models.

```{r}
#| eval: false
library(ForesToolboxRS)
dir.create("./data/testdata")
download.file("https://github.com/ytarazona/ft_data/raw/main/data/LC08_232066_20190727_SR.zip",destfile = "testdata/LC08_232066_20190727_SR.zip")
unzip("testdata/LC08_232066_20190727_SR.zip", exdir = "testdata") download.file("https://github.com/ytarazona/ft_data/raw/main/data/signatures.zip", destfile = "testdata/signatures.zip")
unzip("testdata/signatures.zip", exdir = "testdata")

image <- stack("./data/testdata/LC08_232066_20190727_SR.tif")
sig <- read_sf("./data/testdata/signatures.shp")
classRF <- mla(img = image, model = "randomForest", endm = sig, training_split = 80)
print(classRF)
```

```{r}
#| eval: false
# Classification
colmap <- c("#0000FF","#228B22","#FF1493", "#00FF00")
plot(classRF$Classification, main = "RandomForest Classification", col = colmap, axes = TRUE)
```

![](data/02-lulc-classification/figure-html/unnamed-chunk-3-1.png) \##### Figure
2: LULC map classified with randomForest classifier kernel

```{r}
#| eval: false
#| # Calibration result
plot(
  cal_ml$svm_mccv,
  main = "Monte Carlo Cross-Validation calibration",
  col = "darkmagenta",
  type = "b",
  ylim = c(0, 0.4),
  ylab = "Error between 0 and 1",
  xlab = "Number of iterations"
)
lines(cal_ml$randomForest_mccv, col = "red", type = "b")
lines(cal_ml$naiveBayes_mccv, col = "green", type = "b")
lines(cal_ml$knn_mccv, col = "blue", type = "b")
legend(
  "topleft",
  c(
    "Support Vector Machine",
    "Random Forest",
    "Naive Bayes",
    "K-nearest Neighbors"
  ),
  col = c("darkmagenta", "red", "green", "blue"),
  lty = 1,
  cex = 0.7
)
```

![](data/02-lulc-classification/figure-html/unnamed-chunk-5-1.png)

--------------------------------------------------------------------------------

## Runtime snapshot

```{r}
#| eval: true
devtools::session_info()
Sys.getenv()
.libPaths()
```

## Appendix I

Literature review of current Monte Carlo methods used in REDD+ and ART-TREES
projects

| **Parameter** | **Description** |
|-------------------------|-------------------------------------------------------|
| **Keywords** | Monte Carlo simulations |
|  | Biomass estimation |
|  | Carbon stock uncertainty |
|  | REDD+ projects |
|  | Forest carbon accounting |
|  | Allometric uncertainty |
| **Data Sources** | Scopus |
|  | Web of Science |
|  | Google Scholar |
|  | Grey Literature from REDD+ working groups (i.e. UNFCCC, IPCC) |
| **Temporal Window** | 2003–2023 |
| **Focus Areas** | Applications of Monte Carlo simulations in biomass and carbon stock estimations. |
|  | Addressing uncertainty in input data (e.g., allometric equations, plot-level measurements). |
|  | Integration of Monte Carlo methods in REDD+ policy frameworks and carbon accounting. |
| **Inclusion Criteria** | Peer-reviewed articles and high-impact reviews |
|  | Case studies and empirical research involving REDD+ projects. |
|  | Discussions of methodological advancements or critiques of Monte Carlo approaches. |

##### Table 4: Search parameters used in a review of Monte Carlo tools in REDD+

reporting.

| **REDD+ scheme**[^1] | **Monte Carlo applied** | **Region** | **Key Findings** | **Ref** |
|----------------|----------------|----------------|----------------|:-------------:|
| ADD | Uncertainty of SAAB estimate | Rondônia, Brazil | Estimated ± 20% measurement error in SAAB using Monte Carlo simulations; emphasized large trees’ role in biomass. | @brown1995a |
| ADD | AGB Uncertainty | Kenya, Mozambique | Assessed mixed-effects models in estimating mangrove biomass. | @cohen2013a |
| ADD | Blanket uncertainty propagation | Ghana | AGB prediction error \>20%; addressed error propagation from trees to pixels in remote sensing. | @chen2015b |
| ADD | Plot-based uncertainty | New Zealand | Cross-plot variance greatest magnitude of uncertainty | @holdaway2014a |
| JNR | Multi-scale AGB uncertainty modeling | Minnesota, USA | Cross-scale tests showing effects of spatial resolution on AGB uncertainty. | @chen2016a |
| NA | Allometric uncertainty modeling | Panama | Allometric models identified as largest source of biomass estimation error. | @chave2004error |
| ADD | Sampling and allometric uncertainty | Tapajos Nat Forest, Brazil | Significance of allometric models on uncertainty of root biomass, 95% CI, 21 plots. | @keller2001a |
| ADD | Uncertainty of volume estimates | Santa Catarina, Brazil | Negligible effects of residual uncertainty on large-area estimates | @mcroberts2015a |
| NA | Uncertainty metrics in model selection | Oregon, USA | Uncertainty estimates call for local validation or new local model development | @melson2011a |
| ADD | AGB model uncertainty | French Guiana | AGB sub-model errors dominate uncertainty; height and wood-specific gravity errors are minor but can cause bias. | @molto2013a |
| IFM | Emission factor uncertainty | Central Africa | Model selection is the largest error source (40%); weighting models reduces uncertainty in emission factors. | @picard2015a |
| NA | Uncertainty in ecosystem nutrient estimate | New Hampshire, USA | Identified 8% uncertainty in nitrogen budgets, mainly from plot variability (6%) and allometric errors (5%). | @yanai2010a |

[^1]: ADD: Avoided Deforestation and Degradation, JNR: Jurisdictional & Nested
    REDD+, IFM: Improved Forest Management

##### Table 5: Results of a review of literature on Monte Carlo methodologies in REDD+ projects
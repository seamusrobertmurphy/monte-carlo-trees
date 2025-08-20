pdf_document: 
  highlight: pygments
toc: true
toc_depth: 3
latex_engine: xelatex

bibliography: references.bib
csl: american-chemical-society.csl


Discrete Distributions:
  
  Discrete distributions describe data that can only take on distinct, separate values—often counts, binary (TRUE/FALSE), logical variables, or classification categories.

Bernoulli Distribution: 

Binomial Distribution: 

Poisson Distribution: 

Geometric Distribution: Models the number of failures before the first success in a series of independent Bernoulli trials. For example, the number of sales calls made before securing a sale.

Negative Binomial Distribution: Extends the geometric distribution to model the number of failures before achieving a specified number of successes. For example, the number of trials needed to achieve a fixed number of successful sales, where events exhibit overdispersion compared to a Poisson model.

Discrete Uniform Distribution: Assumes all outcomes in a finite set are equally likely. For example, the outcomes of rolling a fair die.

Continuous Distributions:
  
  Continuous distributions describe data that can take any value within a given range.

Normal (Gaussian) Distribution: A symmetrical, bell-shaped distribution commonly found in nature and statistics. For example, human height measurements.

Lognormal Distribution: Represents data that is right-skewed, where the logarithm of the data follows a normal distribution. Examples include income distribution or stock prices.

Exponential Distribution: Describes the time between events in a Poisson process, where events occur continuously and independently at a constant average rate. For example, the time between customer arrivals or the lifespan of an electronic device.

Uniform (Continuous) Distribution: Assumes all values within a given range are equally likely. For example, a random number generator.

Chi-Square Distribution: Primarily used in hypothesis testing, particularly for goodness-of-fit tests and tests of independence.

t-Distribution (Student’s t): Applied when dealing with small sample sizes or when the population standard deviation is unknown.

Weibull Distribution: Often used in reliability analysis to model the lifespan of products or systems.

Gamma Distribution: Used to model skewed data and is applicable in various technical fields.

Data Analysis Steps Prior to Monte Carlo Simulations

Before conducting Monte Carlo simulations, a thorough examination of the input data’s empirical distributions is essential. This process serves to characterize the data and informs subsequent modeling decisions. The following steps are recommended:
  
  Descriptive Statistics and Distribution Assessment: Begin by calculating key descriptive statistics such as the mean, median, standard deviation, skewness, and kurtosis to summarize the data's central tendency and dispersion. Use visual diagnostic tools to assess the shape of the distributions, including:

Histograms: To visualize the frequency distribution of the data.

Box plots: To identify outliers and assess the data’s spread and symmetry.

Kernel density plots: To provide a smoothed estimate of the probability density function.

Quantile-Quantile (Q-Q) plots: To compare the empirical distribution to a theoretical distribution (such as normal), highlighting any deviations from expected patterns.

Normality Tests: Conduct formal normality tests such as the Shapiro-Wilk test (preferred for smaller samples) or the Kolmogorov-Smirnov test (appropriate for larger samples). These tests provide objective measures of how well the data fits a normal distribution.

Bias Detection and Evaluation: Visualizations, particularly histograms and Q-Q plots, are crucial for identifying potential biases in the data. Compare the empirical distributions to expected or reference distributions to detect systematic deviations. Document any observed biases and assess their potential impact on subsequent calculations and emission factor estimates. These findings can help auditors evaluate how well the proponent addresses and mitigates bias throughout the analysis.

Monte Carlo Simulation Parameterization: The results of the distribution assessment are vital for selecting appropriate probability distributions for the input variables in the Monte Carlo simulation. The chosen distributions should accurately reflect the empirical characteristics of the data (e.g., skewness, kurtosis, modality). Use the descriptive statistics and results from normality tests to parameterize the selected distributions. For software like SimVoi, these empirical distributions should guide the selection of the appropriate Monte Carlo function and the parameters used within that function. Additionally, resampling regimes should be chosen based on the data’s distributions to minimize uncertainty.

Univariate Distribution Summary: Provide a summary of relevant univariate distributions, including:

Statistical names (e.g., normal, lognormal, gamma, Poisson, binomial).

Mathematical definitions and properties.

Typical applications and use cases.

Parameters needed to define each distribution.

Guidance on when each distribution is appropriate based on the data characteristics.

Discrete Distributions:

These distributions deal with data that can only take on distinct, separate values, often counts, TRUE/FALSE or logical variables, and multiple or single strata classifications.  

Bernoulli Distribution: Describes the probability of a single binary outcome such as between success/failure or yes/no responses. For example, flipping a coin once.

Binomial Distribution: Describes the probability of a certain number of successes in a fixed number of independent Bernoulli trials, such as the number of heads in 10 coin flips. 

Poisson Distribution: Describes the probability of a certain number of events occurring within a fixed interval of time or space. For instance, the number of customer arrivals at a store in an hour.

Geometric Distribution: Describes the probability of the number of failures before the first success in a series of independent Bernoulli trials. This could inlcude the number of attempts before a successful sale. 

Discrete Uniform Distribution: Describes a situation where all outcomes are equally likely, and the number of outcomes are finite, such as when rolling a fair dice.

Continuous Distributions:

These distributions deal with data that can take on any value within a given range.  

Normal or Gaussian Distribution: A symmetrical, bell-shaped, assummedly random distribution that is common in nature and statistics. An example would be the heights of people.   

Lognormal Distribution: Describes data that is skewed to the right, where the logarithm of the data follows a normal distribution. Often income distribution and stock prices present this kind of distribution. 

Exponential Distribution: Describes the time between events in a Poisson process, which may be used to distinguish events occurring continuously and independently at a constant average rate, such as the patterns time lengths between customer arrivals, or in the lifespan records of electronic devices. 

Uniform or Continuous Distribution: Describes data where all values within a given range are equally likely. This could include a random number generator. 

Chi-Square Distribution: Used in hypothesis testing, particularly for goodness-of-fit tests and tests of independence. t-Distribution (Student's t-distribution): Used when dealing with small sample sizes and unknown population standard deviations.

Weibull Distribution: Used in reliability analysis to model the lifespan of products or systems. 

Gamma Distribution: Used to model skewed data, and is used in a large variety of technical fields.

Prior to conducting Monte Carlo simulations, a thorough examination of the input data's empirical distributions is essential. This preliminary analysis serves to characterize the data and inform subsequent modeling decisions. Specifically, the following steps are recommended:

Prior to conducting Monte Carlo simulations, a thorough examination of the input data's empirical distributions is essential. This preliminary analysis serves to characterize the data and inform subsequent modeling decisions. Specifically, the following steps are recommended:
  
  Descriptive Statistics and Distribution Assessment: Calculate key descriptive statistics (e.g., mean, median, standard deviation, skewness, kurtosis) to summarize the data's central tendency and dispersion. Employ visual diagnostic tools to assess the shape of the distributions. These should include: Histograms: To visualize the frequency distribution of data. Box plots: To identify outliers and assess data spread and symmetry. Kernel density plots: To provide a smoothed estimate of the probability density function. Quantile-Quantile (Q-Q) plots: To compare the empirical distribution to a theoretical distribution (e.g., normal), highlighting deviations from expected patterns. Conduct formal normality tests, such as the Shapiro-Wilk test (preferred for smaller sample sizes) or the Kolmogorov-Smirnov test (suitable for larger samples), to statistically evaluate deviations from a normal distribution. These tests provide objective measures of goodness-of-fit. Bias Detection and Evaluation: Visualizations, particularly histograms and Q-Q plots, are crucial for identifying potential biases in the data. Compare the empirical distributions to expected or reference distributions to detect systematic deviations. Document any observed biases and assess their potential impact on subsequent calculations and emission factor estimates. The auditor can use these visual and statistical outputs to monitor how effectively the proponent addresses and mitigates bias throughout the analysis. Monte Carlo Simulation Parameterization: The results of the distribution assessment are critical for selecting appropriate probability distributions for the input variables in the Monte Carlo simulation. The chosen distributions should accurately reflect the empirical characteristics of the data (e.g., skewness, kurtosis, modality). Use the calculated descriptive statistics and the results of the goodness-of-fit tests to parameterize the selected distributions. When using software like SimVoi, the empirical distributions need to inform the selection of the correct Monte Carlo function, and the parameters that are used within that function. Resampling regimes should also be selected based upon the data distributions, in order to minimize uncertainty. Univariate Distribution Summary: Provide a comprehensive summary of relevant univariate distributions, including: Statistical names (e.g., normal, lognormal, gamma, Poisson, binomial). Mathematical definitions and properties. Typical applications and use cases. Parameters needed to define the distribution. Guidance on when each distribution is appropriate based on data characteristics. By following these steps, a more robust and statistically sound Monte Carlo simulation can be developed, leading to more often to substantial reductions in uncertainty estimates.
  
  
  Distribution Type

Description

Example

Bernoulli Distribution

Models the probability of a binary outcome (e.g., success/failure or yes/no responses).

Flipping a coin once.

Binomial Distribution

Describes the probability of a specific number of successes in a fixed number of independent Bernoulli trials.

The number of heads in 10 coin flips.

Poisson Distribution

Represents the probability of a given number of events occurring within a fixed interval (time or space).

The number of customer arrivals at a store in an hour.

Geometric Distribution

Models the number of failures before the first success in a series of independent Bernoulli trials.

The number of sales calls made before securing a sale.

Negative Binomial Distribution

Extends the geometric distribution to model the number of failures before achieving a specified number of successes.

The number of trials needed to achieve a fixed number of successful sales, where events exhibit overdispersion compared to a Poisson model.

Discrete Uniform Distribution

Assumes all outcomes in a finite set are equally likely.

The outcomes of rolling a fair die.

Normal (Gaussian) Distribution

A symmetrical, bell-shaped distribution commonly found in nature and statistics.

Human height measurements.

Lognormal Distribution

Represents data that is right-skewed, where the logarithm of the data follows a normal distribution.

Income distribution or stock prices.

Exponential Distribution

Describes the time between events in a Poisson process, where events occur continuously and independently at a constant average rate.

The time between customer arrivals or the lifespan of an electronic device.

Uniform (Continuous) Distribution

Assumes all values within a given range are equally likely.

A random number generator.

Chi-Square Distribution

Primarily used in hypothesis testing, particularly for goodness-of-fit tests and tests of independence.

Used in hypothesis testing (e.g., goodness-of-fit tests).

t-Distribution

Applied when dealing with small sample sizes or when the population standard deviation is unknown.

When dealing with small sample sizes or unknown population standard deviations.

Weibull Distribution

Often used in reliability analysis to model the lifespan of products or systems.

Modeling the lifespan of products or systems.

Gamma Distribution

Used to model skewed data and is applicable in various technical fields.

Modeling skewed data in technical fields.

Bernoulli Distribution

Models the probability of a binary outcome (e.g., success/failure or yes/no responses).

Flipping a coin once.



### Distrobitions

# Bernoulli: binary outcome, where size=1 means one trial.
set.seed(1)
bernoulli_samples <- rbinom(n = 1000, size = 1, prob = 0.5)
table(bernoulli_samples)

# Binomial: number of successes in n independent Bernoulli trials.
set.seed(1)
binomial_samples <- rbinom(n = 1000, size = 10, prob = 0.5)
hist(binomial_samples, main = "Binomial (n=10, p=0.5)", xlab = "Number of successes")

# Poisson: counts the number of events in a fixed interval.
set.seed(1)
poisson_samples <- rpois(n = 1000, lambda = 3)
hist(poisson_samples, main = "Poisson (lambda = 3)", xlab = "Event count")

# Geometric: number of failures until the first success.
# rgeom() returns the count of failures (0 means immediate success).
set.seed(1)
geometric_samples <- rgeom(n = 1000, prob = 0.2)
hist(geometric_samples, main = "Geometric (p = 0.2)", xlab = "Failures until success")

# Negative Binomial: number of failures until r successes occur.
# 'size' is the target number of successes
set.seed(1)
nbinom_samples <- rnbinom(n = 1000, size = 5, prob = 0.3)
hist(nbinom_samples, main = "Negative Binomial (size = 5, p = 0.3)", xlab = "Failures until 5 successes")

# Discrete Uniform: all outcomes in a finite set are equally likely.
set.seed(1)
discrete_uniform_samples <- sample(1:6, size = 1000, replace = TRUE)
table(discrete_uniform_samples)

# Normal: symmetric bell curve.
set.seed(1)
normal_samples <- rnorm(n = 1000, mean = 0, sd = 1)
hist(normal_samples, main = "Normal (mean=0, sd=1)", xlab = "Value")

# Lognormal: right-skewed; log(variable) ~ Normal.
set.seed(1)
lognormal_samples <- rlnorm(n = 1000, meanlog = 0, sdlog = 1)
hist(lognormal_samples, main = "Lognormal (meanlog=0, sdlog=1)", xlab = "Value")

# Exponential: models the time between Poisson events.
set.seed(1)
exponential_samples <- rexp(n = 1000, rate = 1)
hist(exponential_samples, main = "Exponential (rate = 1)", xlab = "Time")

# Continuous Uniform: all values between a and b are equally likely.
set.seed(1)
uniform_samples <- runif(n = 1000, min = 0, max = 1)
hist(uniform_samples, main = "Uniform (0,1)", xlab = "Value")

# Chi-Square: used in hypothesis tests, with df degrees of freedom.
set.seed(1)
chisq_samples <- rchisq(n = 1000, df = 3)
hist(chisq_samples, main = "Chi-Square (df=3)", xlab = "Value")

# t-Distribution: used when sample sizes are small.
set.seed(1)
t_samples <- rt(n = 1000, df = 10)
hist(t_samples, main = "t-Distribution (df=10)", xlab = "Value")

# Weibull: commonly used for reliability or lifespan data.
set.seed(1)
weibull_samples <- rweibull(n = 1000, shape = 2, scale = 1)
hist(weibull_samples, main = "Weibull (shape=2, scale=1)", xlab = "Value")

# Gamma: models skewed data.
set.seed(1)
gamma_samples <- rgamma(n = 1000, shape = 2, rate = 1)
hist(gamma_samples, main = "Gamma (shape=2, rate=1)", xlab = "Value")


When preparing for Monte Carlo simulations, it is good practice first to examine descriptive statistics of the data to characterize the empirical distributions of input variables. This preliminary analysis should include statistical tests of normality, along with visualizations of univariate distributions. Recommended visualizations include histograms, kernel density plots, and Q-Q plots. Together, these tools provide insights into the data's shape, spread, symmetry, skewness, and potential outliers. This appears a seemingly small component but in fact can hold significant material influence on uncertainty values that may lead to increased returns, particularly in project landscapes exhibiting non-normal distributions. 

Accurately characterizing data distributions helps in identifying biases, ensuring data quality, and enhancing confidence in subsequent emissions and biomass estimations. Proper selection of statistical distributions informed by this exploratory analysis significantly improves the reliability and precision of Monte Carlo simulations. Consequently, this reduces uncertainty in estimates of forest biomass and carbon emissions, thereby strengthening the credibility of jurisdictional claims under REDD+ programs and enhancing potential financial returns for Guyana from carbon financing initiatives.

Additionally, these univariate distribution diagrams will greatly help auditors as effective diagnostic resources enabling quicker confirmation and caracterisation of bias that is expected in some biomass data. In this way, these visualisations should may also serve as useful tools in auditors' subsequent assessments of the technical measures and statstical approaches taken by the project to monitor and manage its uncertainty (ART 2021: 8). Winrock stongly recommends incorporating a distribution analysis early in the project's quantitative designs and across its technical SOPs, as we consider this a low hanging fruit of potentially significant impact in either limiting the number of future findings or reduction uncertainty with potential value of -$$$$$$$-. Specifically he project's multiple years of VVB interactions,    impact to selecting the appropriate functions in SimVoi, ensuring more accurate Monte Carlo estimates. In effect, bias corrections are incorporated, reducing uncertainty in the final results and improving confidence in the jurisdiction's claims of nationwide emissions reductions.

When preparing for Monte Carlo simulations, it is best practice to start by examining descriptive statistics to characterize the empirical distributions of input variables. This preliminary analysis typically includes statistical tests for normality and visualizations of univariate distributions, such as histograms, kernel density plots, and Q-Q plots. Together, these tools provide critical insights into the shape, spread, symmetry, skewness, and presence of potential outliers in the data. Although this preliminary step may seem minor, it substantially influences uncertainty estimates, which can directly translate into increased financial returns, particularly within forest project landscapes exhibiting non-normal data distributions.

Accurately characterizing data distributions also helps in identifying and addressing biases, thereby ensuring high data quality and increasing confidence in subsequent estimations of biomass and carbon emissions. Selecting appropriate statistical distributions, informed by exploratory analyses, significantly enhances the reliability and precision of Monte Carlo simulations. Consequently, such careful statistical characterizations reduce overall uncertainty in forest biomass and emissions estimates. In turn, this strengthens the credibility of jurisdictional claims made under REDD+ programs and maximizes potential financial returns for Guyana from carbon financing initiatives.

Univariate distribution visualizations additionally provide auditors with powerful diagnostic resources, enabling rapid identification and characterization of biases commonly encountered in biomass data. These diagrams help auditors efficiently assess the technical rigor and statistical approaches implemented by the project to monitor and manage uncertainty (ART, 2021: 8). Winrock strongly recommends incorporating detailed distribution analyses early in a project's quantitative planning and throughout its technical standard operating procedures (SOPs). Such early integration represents a cost-effective strategy with significant potential for reducing future audit findings, lowering uncertainty, and enhancing financial outcomes for Guyana's REDD+ activities. Specifically, early attention to data distributions directly informs the selection of appropriate simulation functions in tools such as SimVoi, facilitating precise Monte Carlo estimates and robust bias corrections. Ultimately, these enhancements improve confidence in reported nationwide emission reductions.

When preparing for Monte Carlo simulations, it is best practice to start by examining descriptive statistics to characterize the empirical distributions of input variables. This preliminary analysis typically includes statistical tests for normality and visualizations of univariate distributions, such as histograms, kernel density plots, and Q-Q plots. Together, these tools provide critical insights into the shape, spread, symmetry, skewness, and presence of potential outliers in the data. Although this preliminary step may seem minor, it substantially influences uncertainty estimates, which can directly translate into increased financial returns, particularly within forest project landscapes exhibiting non-normal data distributions.

Accurately characterizing data distributions also helps in identifying and addressing biases, thereby ensuring high data quality and increasing confidence in subsequent estimations of biomass and carbon emissions. Selecting appropriate statistical distributions, informed by exploratory analyses, significantly enhances the reliability and precision of Monte Carlo simulations. Consequently, such careful statistical characterizations reduce overall uncertainty in forest biomass and emissions estimates. In turn, this strengthens the credibility of jurisdictional claims made under REDD+ programs and maximizes potential financial returns for Guyana from carbon financing initiatives.



Cont. Distributions

Description of statistical criteria and common use cases 

Bernoulli

Probability of a binary outcome with two possible results, such as success/failure, true/false, yes/no. E.g Probability of getting heads when flipping a single coin.

Binomial

Describes the probability of specific number of successes occurring within fixed set of independent Bernoulli trials. E.g Number of heads in 10 coin flips.

Poisson

Probability of count data, #no. of occurrences of independent events occurring within fixed time period or space. E.g #no. of customers arriving at a store per hour.

Geometric

# of failures until first success. E.g., calls until a sale.

Neg. Binomial

# of failures until r succeeds (overdispersed Poisson).

Discrete Uniform

All finite outcomes equally likely. E.g., rolling a fair die.

Normal (Gaussian)

Symmetrical “bell curve.” E.g., human heights.

Lognormal

Right-skewed; log(variable) ~ Normal. E.g., incomes.

Exponential

Time between Poisson events. E.g., arrival times.

Continuous Uniform

All values in [a,b] equally likely. E.g., random number gen.

Chi-Square

Used in hypothesis tests (e.g., goodness-of-fit).

t-Distribution

Small samples, unknown population SD.

Weibull

Reliability or lifespans.

Gamma

Models skewed data, e.g., wait times.



Discrete Distributions 

Descriptions

Bernoulli

Binary outcome (success/failure). E.g., a single coin flip.

Binomial

# of successes in n Bernoulli trials. E.g., heads in 10 flips.

Poisson

# of events in a fixed interval. E.g., arrivals per hour.

Geometric

# of failures until first success. E.g., calls until a sale.


# --- Replicating SimVoi --- 
# Custom function to simulate from each row (assuming truncnormal)
simulate_truncnorm_from_summary <- function(mean_val, sd_val, min_val = 0, max_val = Inf, n_draws = 10000)
{draws <- truncnorm::rtruncnorm(
  n = n_draws,
  a = min_val,
  b = max_val,
  mean = mean_val,
  sd = sd_val)
return(draws)
}

simulate_truncnorm_from_summary <- function(
    mean_val, sd_val, min_val=0, max_val=Inf, 
    n_draws=10000) {
  draws <- truncnorm::rtruncnorm(
    n     = n_draws,
    a     = min_val,
    b     = max_val,
    mean  = mean_val,
    sd    = sd_val
  )
  # Return vector of draws
  return(draws)
}

# Repeat for AG_Tree
ag_tree_stats <- CarbonStocks_stats %>% filter(Pool == "AG_Tree")
AG_mean <- ag_tree_stats$`mean of all plots (calculated)`
AG_sd   <- ag_tree_stats$`std. dev`
AG_min  <- ag_tree_stats$minimum
AG_max  <- ag_tree_stats$maximum

# We may vote to do a = 0 if we never allow negative carbon:
AG_draws <- simulate_truncnorm_from_summary(
  mean_val = AG_mean, 
  sd_val   = AG_sd, 
  min_val  = 0, 
  max_val  = Inf, 
  n_draws  = 10000)

# Compare results:
mean(AG_draws)
sd(AG_draws)
min(AG_draws)
max(AG_draws)
quantile(AG_draws, probs = c(0.05, 0.95))


# Quick histogram of the draws
hist(AG_draws, breaks=40, col="skyblue", 
     main="Truncated Normal draws for AG Tree",
     xlab="AG Tree (tC/ha)")

# If you want to do this for each carbon pool in a loop, 
# you can add a small function:

simulate_all_pools <- function(df, n_draws=10000) {
  # df is your cs_stats data frame
  # Return a named list of random draws
  # out <- list()
  # for (i in seq_len(nrow(df))) {
  # rowi <- df[i, ]
  # pool_name <- rowi$Pool
  # mean_val  <- rowi$`mean of all plots (calculated)`
  # sd_val    <- rowi$`std. dev`
  # Use zero for min bound; or rowi$minimum if you want to
  # replicate the workbook min
  # draws <- rtruncnorm(
  # n=n_draws,
  # a=0, 
  # b=Inf,
  # mean=mean_val,
  # sd=sd_val
  # )
  # out[[pool_name]] <- draws
  # }
  # return(out)
  # }
  
  all_draws <- simulate_all_pools(CarbonStocks_st_stats, n_draws=10000)
  
  ggplot(data.frame(AG_draws), aes(x = AG_draws)) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = "skyblue", alpha = 0.7) +
    geom_density(col = "red") +
    labs(title = "Monte Carlo Simulation of AG Tree Carbon Pool",
         x = "Carbon Stock (tC/ha)", y = "Density")
  
  
  
  # Function to plot distribution comparisons for a single numeric vector
  plotDistributionComparison <- function(
    CarbonStocks, var_name = "Variable", 
    bw_method = "nrd0", 
    candidate_dists = c("normal", "gamma", "lognormal", "weibull"),
    nbins = 30) {
    
    # Remove missing values
    x <- na.omit(x)
    if(length(x) < 3) {
      warning(paste("Not enough data in", var_name, "to perform analysis."))
      return(NULL)
    }
    
    # Set up a plot using MASS-truehist
    truehist(CarbonStocks, nbins = nbins, xlab = var_name, main = paste("Distribution of", var_name), col="gray")
    # Calculate and overlay a kernel density estimate with the specified bandwidth method
    kd <- density(x, bw = bw_method)
    lines(kd, col = "blue", lwd = 2)
    # Prepare a sequence for plotting fitted densities
    x_seq <- seq(min(x), max(x), length.out = 200)
    
    # Initialize vectors for building the legend
    legend_labels <- c("Kernel Density")
    legend_colors <- c("blue")
    legend_lty <- c(1)
    
    # Fit and plot a Normal distribution if requested
    if("normal" %in% candidate_dists) {
      fit_norm <- try(fitdistr(x, "normal"), silent = TRUE)
      if(!inherits(fit_norm, "try-error")) {
        dens_norm <- dnorm(x_seq, mean = fit_norm$estimate["mean"], sd = fit_norm$estimate["sd"])
        lines(x_seq, dens_norm, col = "red", lwd = 2, lty = 2)
        legend_labels <- c(legend_labels, "Normal Fit")
        legend_colors <- c(legend_colors, "red")
        legend_lty <- c(legend_lty, 2)
      }
    }
    
    # Fit and plot a Gamma distribution (only if all values > 0)
    if("gamma" %in% candidate_dists && all(x > 0)) {
      fit_gamma <- try(fitdistr(x, "gamma"), silent = TRUE)
      if(!inherits(fit_gamma, "try-error")) {
        dens_gamma <- dgamma(x_seq, shape = fit_gamma$estimate["shape"], rate = fit_gamma$estimate["rate"])
        lines(x_seq, dens_gamma, col = "green", lwd = 2, lty = 3)
        legend_labels <- c(legend_labels, "Gamma Fit")
        legend_colors <- c(legend_colors, "green")
        legend_lty <- c(legend_lty, 3)
      }
    }
    
    # Fit and plot a Lognormal distribution (only if all values > 0)
    if("lognormal" %in% candidate_dists && all(x > 0)) {
      fit_lnorm <- try(fitdistr(x, "lognormal"), silent = TRUE)
      if(!inherits(fit_lnorm, "try-error")) {
        dens_lnorm <- dlnorm(x_seq, meanlog = fit_lnorm$estimate["meanlog"], sdlog = fit_lnorm$estimate["sdlog"])
        lines(x_seq, dens_lnorm, col = "purple", lwd = 2, lty = 4)
        legend_labels <- c(legend_labels, "Lognormal Fit")
        legend_colors <- c(legend_colors, "purple")
        legend_lty <- c(legend_lty, 4)
      }
    }
    
    # Fit and plot a Weibull distribution (only if all values > 0)
    if("weibull" %in% candidate_dists && all(x > 0)) {
      fit_weibull <- try(fitdistr(x, "weibull"), silent = TRUE)
      if(!inherits(fit_weibull, "try-error")) {
        dens_weibull <- dweibull(x_seq, shape = fit_weibull$estimate["shape"], scale = fit_weibull$estimate["scale"])
        lines(x_seq, dens_weibull, col = "orange", lwd = 2, lty = 5)
        legend_labels <- c(legend_labels, "Weibull Fit")
        legend_colors <- c(legend_colors, "orange")
        legend_lty <- c(legend_lty, 5)
      }
    }
    
    # Add legend to the plot
    legend("topright", legend = legend_labels, col = legend_colors, lwd = 2, lty = legend_lty, bty = "n")
  }
  
  # Function to loop through all numeric variables in a data frame
  exploratoryMASSAnalysis <- function(data, bw_method = "nrd0", 
                                      candidate_dists = c("normal", "gamma", "lognormal", "weibull"),
                                      nbins = 30) {
    # Identify numeric columns
    num_vars <- names(data)[sapply(data, is.numeric)]
    
    # Set up a multi-panel plotting layout (adjust rows/columns as needed)
    n <- length(num_vars)
    ncol <- 2
    nrow <- ceiling(n / ncol)
    op <- par(mfrow = c(nrow, ncol))
    
    # Loop over each numeric variable and generate plots
    for (var in num_vars) {
      plotDistributionComparison(data[[var]], var_name = var, bw_method = bw_method,
                                 candidate_dists = candidate_dists, nbins = nbins)
    }
    
    # Reset plotting layout
    par(op)
  }
  
  
  
# --- Tody DaTA --- 
  CarbonStocks = CarbonStocks |> 
    dplyr::rename(Statistic = x1)|>
    select(
      `Statistic`           = 1,
      `AG_Tree`             = 2,
      `BG_Tree`             = 3,
      `Saplings`            = 4,
      `StandingDeadWood`    = 5,
      `LyingDeadWood`       = 6,
      `SumCarbonNoLitter`   = 7,
      `Litter`              = 8,
      `SumCpoolWLitter`     = 9,
      `SumCO2e`             = 10,
      `Soil_tC_ha`          = 11,
      `SumALL_POOLS_CO2eha` = 12,
      `SumABGBLiveTree`     = 13
    ) %>% slice(1:9)
  
  # Convert wide to long, use "Statistic" to define row
  CarbonStocks_long <- CarbonStocks |>
    tidyr::pivot_longer(
      cols = -Statistic,
      names_to = "Pool",
      values_to = "Value"
    ) |> mutate(Value = as.numeric(Value))
  
  # Convert from long back to wide format:
  CarbonStocks_wide <- CarbonStocks_long %>%
    pivot_wider(
      names_from = Statistic,
      values_from = Value)
  
  # Transpose to long dataframe: flipping rows w/ columns
  CarbonStocks_long <- CarbonStocks |>
    tidyr::pivot_longer(
      cols = -Statistic,
      names_to = "Pool",
      values_to = "Value"
    ) |> mutate(Value = as.numeric(Value))
  
  # Pivot back to wide dataframe & “Statistic” becomes a row:
  CarbonStocks_wide <- CarbonStocks_long %>%
    pivot_wider(
      names_from = Statistic,
      values_from = Value)
  
  
  

  
  ###### *Table 1: Continuous data distributions, and example use cases for Monte Carlo simulations.*
  
  | Distribution | Statistical Criteria & Use Cases | PDF |
    |---------------------------------|------------------------------------------------|-------------------------------------------------|
    | Normal (Gaussian) | Symmetric, bell-shaped distribution used for modeling continuous variables: biomass/ha | $\displaystyle \begin{aligned} f(x)&=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \end{aligned}$ |
    | Lognormal | Right-skewed distribution suitable for variables constrained to positive values (e.g., emission rates). | $\displaystyle f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)$ |
    | Exponential | Models waiting times between independent events, such as forest fire occurrences or logging events. | $\displaystyle \begin{aligned} f(x)&=\lambda e^{-\lambda x},\\[3pt] &\quad x\ge0 \end{aligned}$ |
    | Continuous Uniform | Assumes all values in an interval [a, b] are equally likely; useful for random spatial sampling in forests. | $\displaystyle \begin{aligned} f(x)&=\frac{1}{b-a},\\[3pt] &\quad a\le x\le b \end{aligned}$ |
    | Chi-Square | Often used in goodness-of-fit tests to evaluate model accuracy in biomass estimation. | $\displaystyle f(x)=\frac{1}{2^{k/2}\Gamma(k/2)}\,x^{\frac{k}{2}-1}e^{-x/2},\quad x>0$ |
    | t-Distribution | Suitable for small sample sizes with unknown population stdev (e.g., limited forest carbon data). | $\displaystyle \begin{aligned} f(x)&=\frac{\Gamma\left(\frac{v+1}{2}\right)}{\sqrt{v\pi}\,\Gamma\left(\frac{v}{2}\right)}\left(1+\frac{x^2}{v}\right)^{-\frac{v+1}{2}} \end{aligned}$ |
    | Gamma | Models positively skewed data, such as biomass growth rates or carbon accumulation over time. | $\displaystyle f(x)=\frac{x^{k-1}e^{-x/\theta}}{\theta^k\Gamma(k)}$ |
    | Weibull | Flexible distribution used in reliability analysis, e.g., modeling tree mortality. | $\displaystyle \begin{aligned} f(x)&=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k} \end{aligned}$ |
    
    ###### *Table 2: Discrete data distributions, and example use cases designed with Monte Carlo simulations.*
    
    | Distribution | Statistical Criteria & Use Cases | PMF |
    |---------------------------------|------------------------------------------------|-------------------------------------------------|
    | Bernoulli | Binary outcome probability, e.g., presence/absence of deforestation in an area. | $\displaystyle \begin{aligned} P(X=x)&=p^{x}(1-p)^{1-x},\\[3pt] x&\in\{0,1\} \end{aligned}$ |
    | Binomial | Probability of fixed #no. of successes over $n$ Bernoulli trials, e.g., no. of heads in 10 coin flips. | $\displaystyle \begin{aligned} P(X=k)&=\binom{n}{k}p^{k}(1-p)^{n-k},\\[3pt] k&=0,1,\dots,n \end{aligned}$ |
  | Poisson | Models counts of independent events within an interval, e.g., number of wildfire incidents per year. | $\displaystyle \begin{aligned} P(X=k)&=\frac{\lambda^{k}e^{-\lambda}}{k!},\\[3pt] k&=0,1,2,\dots \end{aligned}$ |
    | Geometric | Models #no. of trials until the first success, e.g., number of inspections until detecting deforestation. | $\displaystyle \begin{aligned} P(X=k)&=(1-p)^{k-1}p,\\[3pt] k&=1,2,\dots \end{aligned}$ |
  | Negative Binomial | Counts #no. failures until $r$ successes occur, treats overdispersed or repeated deforestation detections. | $\displaystyle \begin{aligned} P(X=k)&=\binom{k+r-1}{k}(1-p)^{r}p^{k},\\[3pt] k&=0,1,2,\dots \end{aligned}$ |
  | Discrete Uniform | Assumes outcome in a finite set is equally likely, e.g., random sampling of inventory across a forest. | $\displaystyle \begin{aligned} P(X=x)&=\frac{1}{n},\\[3pt] x&=1,2,\dots,n \end{aligned}$ |
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
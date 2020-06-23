# Problems:

# Lecture 1
### Flow Control:
Consider the following (centered, normalized) tidal data collected from a factories outflow over the course of a 10 hour period centered at 0:00 AM. 
||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|
|Hours|-5.14| -5.08| -2.78| -0.74| -0.0| 0.01| 0.46| 2.68| 3.36| 5.74|
|Meters|-0.76| -0.71| -0.53| -0.16| 0.09| 0.28| 0.18| 0.76| 1.02| 1.0|
The units are hours after 0:00 AM and Meters above mean tide. 

1) First, compute the expression for the linear regression using standard calculus. Lets see that we get the same expression as for KNN. 
*Write out the expression for RSS explicitly and take the two derivative. Then show that it matches the matrix formula*

2) Use both the regression above and k-NN woth k=3 to estimate the tidal height at 2:00, and 4:00.  In each case, which do you trust more?
*The regression line will be more accurate at 2:00 than at 4:00 probably, with k-NN being more accurate at 4:00. Secretly of course we know that the background curve will be roughly sinusoidal.*
3) Should you use k-NN or linear regression to estimate the tidal height at -6:00? What about 0:00?
*k-NN has nothing to say about cases beyond the boundary (check for a couple values of k) regression will be inaccurate, but best at the edges. In the middle though I would expect k-NN to be a better fit since you can average many readings over a local area.*
4) Describe a case where k-NN will almost always do better than regression? 
*Extend the period to 24 hours, that is a full arc of a sine function. Regression will just be a flat line, k-NN will always be better.*

#### Generating Code:

>np.random.seed(7)
>X = (np.sort(np.random.random(10))-.5)*12
>y = np.sin(X/4) + np.random.random(10)*.3
>plt.plot(X,y,'o')

# Lecture 2
### Exercise 1
Assume that $\mathbf{x}$ and $\mathbf{y}$ depend on $\mathbf{z}$. Show that
$$
\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{x}) }{\partial \mathbf{z}} = \frac{\partial \mathbf{y}}{\partial z} A \mathbf{x} + \frac{\partial \mathbf{x}}{\partial z}A^T\mathbf{y}
$$

### Exercise 2
Let $\mathbf{A}$ be a invertible matrix that depends on a scalar $x$. Use the fact that $\mathbf{A}^{-1}\mathbf{A} = I$ to show that

$$
\frac{d \mathbf{A}^{-1} }{d x} = \mathbf{A}^{-1} \frac{d \mathbf{A}}{d x}\mathbf{A}^{-1}
$$


# Lecture 3
Recall the data from the problems in Lecture 1:

Consider the following (centered, normalized) tidal data collected from a factories outflow over the course of a 10 hour period centered at 0:00 AM. 
||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|
|Hours|-5.14| -5.08| -2.78| -0.74| -0.0| 0.01| 0.46| 2.68| 3.36| 5.74|
|Meters|-0.76| -0.71| -0.53| -0.16| 0.09| 0.28| 0.18| 0.76| 1.02| 1.0|
The units are hours after 0:00 AM and Meters above mean tide. 
1) Compute the point estimate for the height of the tide, and its confidence interval under the assumption of normally distributed $y$ values. 
*Here, the point estimate with be the mean and the CI will be calculated as in slides 12-15. You will have to look up the z-values. State how you do that.*
2) Compute the parameter variance for $\beta_0$ and $\beta_1$. 
*Follow slide 23. You should write down what $\mathbf{X}$ is and how you got it but you can use Python/Matlab to do the calculation.*
3) Compute the $z$-score (since we're assuming an actually normal distribution) and confidence intervals for $\beta_0$ and $\beta_1$. Which parameter, if either, is statistically significant at the $\alpha=.05$ level? What does the mean about the average height of the tide vs the speed the tide comes in?
*Follow slides 28 and 29. For what it means, $\beta_0$ corresponds to the linear offset which would be the mean under RSS (note: it will not be under another metric and you should state this explicitly). The overall speed will be roughly estimated by the linear parameter.*
4) Does our assumption that the data is normally distributed hold for the $y$-values alone? What about for the $y$-values as deviants from a line?
*Answer: The first is obviously highly unlikely, since the $y$ values are far from normally distributed. The second is trickier, you should look at the residual plot. You'll discover again that it's unlikely to be normal deviants from a linear classifier.*

#### 
Consider the following inference problem, where $y$ has been generated as a linear combination of the features $X_i$. 
|$y$|$X_1$|$X_2$|$X_3$|
|-|-|-|-|
|-1.22|  0.08|  0.78|  0.44|
|-0.84|  0.72|  0.98|  0.54|
|-0.27|  0.5 |  0.07|  0.27|
|-1.86|  0.5 |  0.68|  0.8 |
|-0.45|  0.38|  0.07|  0.29|
|-0.36|  0.91|  0.21|  0.45|
|-0.79|  0.93|  0.02|  0.6 |
|-0.66|  0.95|  0.23|  0.55|
|-0.6 |  0.91|  0.13|  0.52|
|-0.63|  0.75|  0.67|  0.47|

1) Perform the RSS fit. Compute the z-score (since we have very few data points) for each parameter $\beta_0, \beta_1, \beta_2,\beta_3$. What should we conclude from these $z$-scores?
*You should write the resulting $\mathbf{X}^{-1}\mathbf{X}$ matrix and use it to construct the $z$-score but you certainly don't have to do the matrix multiplication by hand. Show how to extract the $z$-scores from the matrix.  We should probably conclude that $\beta_0=0$.*
2) Compute the $F$-score for dropping $X_2$ and for dropping $X_3$. What does the F score say about which factor contributes least?
#### Generating Code:
> np.random.seed(7)
> X = np.random.random([10,3])
> y = X[:,0] - 3*X[:,2] + np.random.random(10)*.1

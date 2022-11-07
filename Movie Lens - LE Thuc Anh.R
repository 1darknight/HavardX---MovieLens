# ---
# title: "Project Movie Lens Report"
# author: "Le Thuc Anh"
# date: "May 20, 2022"
# ---


# Packages installation ---------------------------------------------------

# Require packages
Packages <- c("data.table", "tidyverse", "ggplot2", "caret", "Matrix", "hrbrthemes", "viridis", "recosystem")
# Print non-scientific number with maximum 4 signifcant digits 
options("scipen" = 999, "digits" = 4)
lapply(Packages, require, character.only = TRUE)


# Data Preparation --------------------------------------------------------

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Naive model ------------------------------------------------------------------

mu = mean(edx$rating)
mu

naive_rmse  = RMSE(validation$rating, mu)
naive_rmse


# The movie effects -------------------------------------------------------

# movie buffer 
mu = mean(edx$rating)
buffer_mv = edx[ , mean(rating - mu), by = movieId][,.(movieId, bf_mv = V1)]

# predict with movie effect
pred_movie_effects = validation[buffer_mv, on = 'movieId', pred_rating := mu + bf_mv ][, pred_rating ]
RMSE(validation$rating, pred_movie_effects)


# The user effects --------------------------------------------------------

# user buffer 
mu = mean(edx$rating)
buffer_us = edx[, mean(rating - mu), by = userId][,.(userId, bf_us = V1)]

# predict with movie effect
pred_us_mv_effects = validation[buffer_mv, on = 'movieId', bf_mv := i.bf_mv][buffer_us, on = 'userId',  bf_us := i.bf_us][, pred := mu + bf_mv + bf_us][, pred]
RMSE(validation$rating, pred_us_mv_effects)


# Regularization ----------------------------------------------------------


# Why we need regularization?

head(validation[, residual := (rating - (mu + bf_mv))][order(-abs(residual))][,title], 10)

movie_titles = unique(edx[,c('movieId', 'title')])

ten_best = head(buffer_mv[movie_titles, on = 'movieId', title := title][order(-bf_mv)][,title], 10)
ten_best

ten_worst = head(buffer_mv[movie_titles, on = 'movieId', title := title][order(bf_mv)][,title], 10)
ten_worst

edx[movie_titles, on = 'movieId', title := title][buffer_mv, on = 'movieId', bf_mv := i.bf_mv][title %in% ten_best ,.N, by = c('movieId', 'bf_mv')][order(-bf_mv), N]

edx[movie_titles, on = 'movieId', title := title][buffer_mv, on = 'movieId', bf_mv := i.bf_mv][title %in% ten_worst ,.N, by = c('movieId', 'bf_mv')][order(bf_mv), N]

# Penalized least square  
## Choosing penalty term (lambda)

lambdas <- seq(0, 10, 0.25)

fn_lmbd_rmse <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  mv <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  us <- edx %>% 
    left_join(mv, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(mv, by = "movieId") %>%
    left_join(us, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})
data.table(l = lambdas, 
           rmses = fn_lmbd_rmse) %>%
  ggplot(aes(x = l, y = rmses)) +
  geom_point() +
  theme_minimal()

lambda = lambdas[which.min(fn_lmbd_rmse)]
print(lambda)

# Regularization
mv <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
us <- edx %>% 
  left_join(mv, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings_reg <- 
  validation %>% 
  left_join(mv, by = "movieId") %>%
  left_join(us, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(predicted_ratings_reg, validation$rating)


# Matrix Factorization ----------------------------------------------------

# transform train data
train_edx <- with(edx, data_memory(user_index = userId,
                                   item_index = movieId,                    
                                   rating = rating))
# transform test data
test_vali <- with(validation, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating)) 
# create model object 
r <-  recosystem::Reco()
# training model 
r$train(train_edx)

# predict with model
y_hat_edx <-  r$predict(test_vali, out_memory())

# RMSE
RMSE(validation$rating, y_hat_edx)


# Results -----------------------------------------------------------------

RMSE_result = data.table(method = c("Naive model", "Movie effect", "User effect", "Regularization", "Matrix Factorization"),
                         RMSE = c(RMSE(validation$rating, mu),
                                  RMSE(validation$rating, pred_movie_effects),
                                  RMSE(validation$rating, pred_us_mv_effects),
                                  RMSE(predicted_ratings_reg, validation$rating),
                                  RMSE(validation$rating, y_hat_edx)
                         ))
RMSE_result[, pct_improved := (((lag(RMSE, 1) - RMSE) / lag(RMSE, 1) )*100)]

print(RMSE_result)
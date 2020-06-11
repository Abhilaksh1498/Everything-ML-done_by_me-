dataset = read.csv('Ads_CTR_Optimisation.csv')
N= 10000
d=10
no_of_selections = integer(d)
sums_of_rewards= integer(d)
ads_selected = integer()
for (n in 1:N){
  if (n<=10){
    ads_selected = append(ads_selected, n)
    no_of_selections[n] = no_of_selections[n] + 1
    sums_of_rewards[n] = sums_of_rewards[n] + dataset[n,n]
  }

  else{
    upper_bound = integer()
    for (i in 1:d){
      upper_bound = append(upper_bound, sums_of_rewards[i]/no_of_selections[i]+ sqrt(1.5*log(n)/no_of_selections[i]))
    }
    next_index = match(max(upper_bound),upper_bound)
    no_of_selections[next_index] = no_of_selections[next_index]+ 1
    sums_of_rewards[next_index]= sums_of_rewards[next_index]+ dataset[n,next_index]
    ads_selected = append(ads_selected, next_index)
  }
}
total_rewards = sum(sums_of_rewards)

hist(ads_selected)

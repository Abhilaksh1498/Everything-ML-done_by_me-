dataset = read.csv('Ads_CTR_Optimisation.csv')
N =10000
d = 10
num_of_reward1 = integer(d)
num_of_reward0 = integer(d)
ads_selected = integer()
for (n in 1:N){
  theta = integer()
  for (i in 1:d){
    theta = append(theta, rbeta(n=1 , shape1 = num_of_reward1[i]+1, shape2 = num_of_reward0[i]+1))
  }
  next_index = match(x=max(theta), table = theta)
  ads_selected = append(ads_selected, next_index)
  if (dataset[n, next_index] == 1){
    num_of_reward1[next_index]= num_of_reward1[next_index]+1 
  }
  else{
    num_of_reward0[next_index] = num_of_reward0[next_index]+1
  }
}
total_reward = sum(num_of_reward1)
hist(ads_selected, c = 'red')
# Campus Battery Agent: speaker notes

Slides: [slides.pdf](slides.pdf) (or open `slides.html` locally to see slides and notes together).

## 1. A battery agent for hard mode

(about 10 s) Hi, we're Fanboy and Chum Chum. We did the hard mode of the UPC Sostenible challenge. I'll explain how our agent works and what we learned.

## 2. The problem

(about 20 s) In hard mode the battery is three times bigger and there is twelve times more solar. And the main difference: we only see the price and CO2 of the next three hours, instead of 24. So we have to predict the rest ourselves: price, CO2, the campus demand and the solar production.

## 3. How good can an agent be?

(about 30 s) The first thing we did was check how good an agent can be. An optimiser that knows the whole future gets 21.26 points on our validation weeks. We measured every version of our agent against that. The first one lost 2.45 points and the final one loses 1.07, so we get 95 percent. Most of what we lose is from not knowing price and CO2 after three hours.

## 4. How the agent works

(about 30 s) This is what the agent does every hour. It predicts the campus demand, the solar and the price and CO2. Then it finds the best plan for the next 36 hours, but it only does the first hour and repeats everything the next hour. Finally it clips the action so it never breaks a battery limit. We tested changes on held-out weeks, like the real test.

## 5. What we predict and how

(about 30 s) For the campus demand we use LightGBM with 28 features: hour and weekday, the last readings, the same hour yesterday and last week, temperature and a holiday indicator, plus a few rules because the meter doesn't update at night. This cut the error in half. For solar we use ridge regression on how clear the sky is. LightGBM was worse there. For price and CO2, ridge regression on the last 48 hours.

## 6. How we decide what the battery does

(about 30 s) Deciding what the battery does is a linear program: we minimise what we import from the grid in the next 36 hours, with the 70/30 weights of the score. We started with scipy's solver and then wrote our own with dynamic programming. It doesn't discretise the battery level, so the result is exactly the same, about twenty times faster. And we don't plan on a single forecast. For demand and solar we plan on several possible values, because if we discharge more than the campus uses, the rest is given away. That was our biggest improvement, 0.71 points.

## 7. What came from us, not from the model

(about 35 s) Two things came from us and not from the model. First, holidays. The model handled holidays in general but not the ones only Barcelona has, and they aren't in the training data, so validation said it wasn't worth it. We know the campus closes on those days, so we added three dates and gained 0.23 on the public leaderboard. Second, the public and private split. We didn't find the answer ourselves, but we decided it was the important question and used submissions to test it.

## 8. Public and private leaderboard

(about 35 s) The public leaderboard is the first half of each test week and the private one is the second half. But it's one simulation with one battery, so energy charged in the public half can be used in the private half. We selected two agents. h4 is general and ignores this. h3p fills the battery just before the private half and uses everything before the week ends, and the code says so. Also, price and CO2 are public data from Red Eléctrica. We told the organisers and didn't use it.

## 9. Summary

(about 20 s) To sum up: we get 95 percent of the best possible score seeing only three hours. On the public test that's 22 percent less CO2 and 26 percent less cost, with no infractions. What's left to improve is the price and CO2 forecast. Thanks! Any questions?

## 10. Backup: how much to discharge

Backup slide, for questions. This is the rule that comes out of the optimiser for the current hour. We discharge one more kilowatt only if what it earns now, times the chance that the campus really uses it, is more than what that energy is worth later. The left plot is a weekday at one in the afternoon, where the forecast is 443 kilowatts. If the energy is worth almost the same later, we only discharge 214, the part we are almost sure will be used. If it's worth much less later, we discharge 523. The right plot shows that at night we just discharge the forecast, and at midday it depends a lot.

## 11. Backup: things we saw in the data

Backup slide, for questions. Things we found in the data. The load meter is not read between eleven at night and five in the morning, so the night demand is known, and at six it jumps. If the solar panels are colder than the air before sunrise, the sky is clear and the day will be sunny. The battery is bigger than what the campus can use: in 87 percent of the hours we discharge, the limit is the demand. Because of that, predicting too little sun costs us and predicting too much almost doesn't. For price and CO2, a more accurate forecast did not improve the score four times. And an agent that only acts in August got zero, so there is no August week in the test.

## 12. Backup: the Barcelona holidays

Backup slide, for questions. Our validation weeks are days 1 to 21 of each month, but the test weeks are the last seven days. Three of those test weeks start on a holiday: La Mercè on 24 September, Segona Pasqua on 25 May and Sant Joan on 24 June. On a holiday the campus uses about half the energy, but the electricity price looks like a normal weekday. So the agent bought cheap energy at night for the morning, and then the campus was closed. We added the three dates to the demand model. We predicted plus 0.20 and got plus 0.23 on the public leaderboard. This is a public gain: the three holidays are in the first half of their week, so the private score is the same with or without them.

## 13. Backup: predicted and real scores

Backup slide, for questions. We didn't tune against the leaderboard. Before each submission we wrote down the score we expected. Here are six examples. Only one was off, v2w, by half a point. The low public scores of h3w and h3p are on purpose: those agents give up the public half to do better in the private half.

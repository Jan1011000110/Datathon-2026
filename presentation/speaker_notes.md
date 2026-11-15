# Campus Battery Agent: speaker notes

Slides: [slides.pdf](slides.pdf) (or open `slides.html` locally to see slides and notes together).

## 1. A battery agent for hard mode

(about 10 s) Hi, we're Fanboy and Chum Chum. We did the hard mode of the UPC Sostenible challenge. I'll explain how our agent works and what we learned.

## 2. The problem

(about 20 s) In hard mode the battery is three times bigger and there is twelve times more solar. And the main difference: we only see the price and CO2 of the next three hours, instead of 24. So we have to predict the rest ourselves: price, CO2, the campus demand and the solar production.

## 3. How good can an agent be?

(about 20 s) First we checked how good an agent can be. An optimiser that knows the whole future gets 21.26 points on our validation weeks. Our first version lost 2.45 points against that and the final one loses 1.07, so we get 95 percent. What we lose is mostly from not knowing price and CO2 after three hours.

## 4. How the agent works

(about 20 s) This is what the agent does every hour. It predicts the campus demand, the solar and the price and CO2. Then it finds the best plan for the next 36 hours, does only the first hour, and repeats. Finally it clips the action so it never breaks a battery limit.

## 5. What we predict and how

(about 30 s) For the campus demand we use LightGBM with 28 features: hour and weekday, the last readings, the same hour yesterday and last week, temperature and a holiday indicator, plus a few rules because the meter doesn't update at night. This cut the error in half. For solar we use ridge regression on how clear the sky is. LightGBM was worse there. For price and CO2, ridge regression on the last 48 hours.

## 6. How we decide what the battery does

(about 30 s) Deciding what the battery does is a linear program: minimise what we import from the grid in the next 36 hours, with the 70/30 weights of the score. We wrote our own solver with dynamic programming: same result as scipy's LP solver, about twenty times faster. And we don't plan on a single forecast. For demand and solar we plan on several possible values. That was our biggest improvement, 0.71 points.

## 7. How much to discharge

(about 20 s) This is the rule the optimiser follows. It discharges one more kilowatt only if what it earns now, times the chance that the campus really uses it, is more than what the energy is worth later. So at midday, when solar is uncertain, it is more careful.

## 8. Things we saw in the data

(about 20 s) Some things we saw in the data. The load meter stops at night, so the night demand is known. Cold panels before sunrise mean a sunny day. And the battery is bigger than what the campus can use, so discharging too much is wasted.

## 9. What came from us, not from the model

(about 20 s) Two things came from us and not from the model. First, the Barcelona holidays: the model handled holidays in general but not the local ones, and validation couldn't see them. Second, the public and private split. We didn't find the answer ourselves, but we decided it was the important question and used submissions to test it.

## 10. The Barcelona holidays

(about 20 s) The holidays in more detail. Three test weeks start on a holiday. The campus uses half the energy but the price looks like a normal weekday, so the agent bought energy for a closed campus. Adding three dates gave 0.23 on the public leaderboard. It's a public gain, not why we're first.

## 11. Predicted and real scores

(about 10 s) Before each submission we wrote down the score we expected. Here are six examples, and only one was off, by half a point. So we knew where the points came from.

## 12. Public and private leaderboard

(about 30 s) The public leaderboard is the first half of each test week and the private one is the second half. But it's one simulation with one battery, so energy charged in the public half can be used in the private half. h4 is our general agent and ignores this. h3p fills the battery just before the private half, and the code says so. Also, price and CO2 are public data. We told the organisers and didn't use it.

## 13. Summary

(about 20 s) To sum up: we get 95 percent of the best possible score seeing only three hours. On the public test that's 22 percent less CO2 and 26 percent less cost, with no infractions. What's left to improve is the price and CO2 forecast. Thanks! Any questions?

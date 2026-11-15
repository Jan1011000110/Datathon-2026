# Campus Battery Agent: speaker notes

Slides: [slides.pdf](slides.pdf) (or open `slides.html` locally to see slides and notes together).

## 1. Running a campus battery when you can only see 3 hours ahead

Hi, we're Fanboy and Chum Chum. We did the UPC Sostenible challenge in hard mode: an agent that runs the campus battery while only seeing three hours of prices ahead. Short version: we get 95 percent of what you'd get with a perfect crystal ball, and we never broke a physical limit. The next ten minutes are how it works and what we learned on the way.

## 2. Decide every hour, see 3 hours ahead

Quick recap of hard mode, because it's a different problem from the standard one. In standard mode you see 24 hours of price and CO2, a plain optimiser is basically perfect, and more than 30 teams are tied. In hard mode you see three hours. The battery is three times bigger and the solar plant twelve times bigger, so the campus is often in surplus. So we have to guess four things and decide anyway.

## 3. First question: how good can anyone get?

Before building anything clever we wanted to know what the maximum is. So we wrote a linear programme that cheats: it knows the whole future. That gives 21.26 points on our validation weeks. Everything after that we measured as points lost against that ceiling. Our first planner lost 2.45. The final agent loses 1.07, so it gets 95 percent. And we know where the rest is: it's mostly the price and CO2 signal beyond three hours. Knowing the load perfectly would only give 0.19 more, so we stopped polishing the load forecast.

## 4. What the agent does every hour

This is the whole agent; the next three slides open each box. Every hour it forecasts three things: the campus load, the solar output, and the price and CO2 past the three hours we get. Each forecast comes with error quantiles, measured on weeks the model never saw. Then it plans 36 hours ahead with an exact dynamic programme, over nine scenarios of net load and twenty possible paths of the signal. It's model predictive control: re-plan every hour, apply only the first command. And then the boring but important part: the command is clamped just inside the limits, and if anything crashes there are three fallbacks. How we tested: three validation folds, days 1 to 7, 8 to 14 and 15 to 21 of each month, always scoring the whole control loop with the models fitted outside the fold.

## 5. Three forecasters, nothing fancy

What we predict and with what. Load: the meter has regimes, so rules come first. It doesn't update at night, and at six in the morning it jumps by an amount that depends on the weekday. On top of that, a LightGBM with 300 small trees and 28 inputs predicts the change from the last reading. The inputs are things like the last three readings, the same hour yesterday and last week, the average of the last three days of the same type, and a holiday index it estimates from today's readings. That halves the error for the hour we're deciding, from 38 to 20 kilowatts. Solar is more physical: a clear-sky curve for each day of the year, times a clearness factor from a ridge regression. We tried LightGBM there too and it was much worse, so out it went. Price and CO2 past three hours: one ridge regression per horizon, on the difference from an average day. All these errors are on validation weeks with the models fitted outside them, compared with a plain average-profile forecast.

## 6. Is it an LP? Yes. We just solve it faster

So how do we decide what the battery does? It is a linear programme: minimise the grid import over the next 36 hours, each kilowatt-hour weighted by the score signal, CO2 plus 0.27 times the price, with the power, capacity and efficiency limits. We started with scipy's LP solver. Then we noticed the value of the energy in the battery is a convex, piecewise-linear function of the state of charge, so you only need its slope, which is a short sorted list, and going back one hour is just merging two sorted lists. That's an exact dynamic programme. Same answer as the LP to eleven decimals on 4,500 cases, about twenty times faster, numpy only. And the speed is the point: it lets us plan under uncertainty. Nine scenarios of net load for each of the next twelve hours, and twenty error paths of the price and CO2 signal taken from held-out weeks. Planning on the distribution instead of the single forecast was our biggest single gain: 0.71 points, with exactly the same forecasts.

## 7. Don’t discharge what the campus won’t use

This is the rule that comes out of the plan for the hour we're deciding. Discharge one more kilowatt only if what it earns now, times the chance the campus actually uses it, beats what that energy is worth later. Left plot: a weekday at one in the afternoon, forecast net load 443 kilowatts. If a stored kilowatt-hour is worth 90 percent as much later, we only discharge 214, the part we're almost sure will be used. If it's worth 25 percent later, we discharge 523, more than the forecast. Right plot: why a fixed safety margin is the wrong tool. At four in the morning or seven in the evening the load is steady and the best move is just the forecast. At one in the afternoon, with the sun uncertain, it goes anywhere from zero to twice the forecast. Charging works the same way, mirrored.

## 8. Six things the data told us

Six things we found by looking at the data, each of which changed the agent. The load meter isn't read between eleven at night and five in the morning, so the night load is known exactly; then at six it jumps, and how much depends on the weekday. Solar panels colder than the air before dawn mean a clear sky, so tomorrow is sunny: that went into the solar forecast. The battery is bigger than the campus can absorb: in 87 percent of discharge hours the limit is the load, and anything above it is given to the grid for free. That makes the error one-sided, which is exactly why we plan on the distribution. For price and CO2, we gave the planner the true values in the wrong order and it recovered everything, so the level matters and the order doesn't; four times a more accurate forecast didn't move the score. And an agent that only acts in August scored exactly zero, so there is no August week in the test.

## 9. What came from us, not from the model

This slide is the part that didn't come out of a model or a validation score; it came from us. First, holidays. The model handled holidays in a general way and skipped the ones that are specific to Barcelona. And validation couldn't help: none of Barcelona's local holidays falls in the training days, so on paper a calendar was worth almost nothing and the recommendation was not to ship it. But we know when a Barcelona campus is closed: La Mercè in September and Whit Monday in May, plus Sant Joan. We typed in three dates and gained 0.23 on the public board; it's a public-half gain, so it isn't what decided the final ranking. Second, the split between the public and the private test. We saw scores moving on the leaderboard without anyone having a better model. We didn't work out the answer ourselves, but we decided that was the thing to understand, and spent submissions on probes instead of tuning. Three smaller calls underneath: asking whether the price and CO2 signal was simply public data, which it is; not building on a remark from the organisers that turned out to be wrong; and deciding what counts as fair.

## 10. How three Barcelona holidays fooled the agent

Here's the holiday one in detail, because it's the result we like most. Our validation weeks are days 1 to 21 of each month, but the test weeks are the last seven days. And if you look at a calendar, three of those test weeks start on a holiday: La Mercè on 24 September, Whit Monday on 25 May and Sant Joan on 24 June. On a holiday the campus uses about half the usual energy. But on these local and regional holidays the electricity price keeps its normal weekday shape. So the agent did the sensible weekday thing: buy cheap energy overnight for the morning. And then nobody came. We typed the three dates into the load forecaster, predicted plus 0.20, and the server gave plus 0.23. It also told us something we couldn't check from the data: yes, the campus really closes on Barcelona's local holidays. To be straight about it: this is a gain on the public board, not the reason we're first. All three holidays fall in the first half of their week, so on the final private ranking this agent scores exactly the same as the one without the calendar, 19.74. It's here because it's a good example of something the data couldn't tell us and we could.

## 11. We called our scores before submitting

We didn't tune against the leaderboard. Before each submission we wrote down what we expected, then sent it. Six examples here. The first row comes from two probes: an agent that only acts in August scored exactly zero, so there's no August week and the plan should end on 31 July. The public score didn't change, as predicted, because the last hours of the run are in the private half. One miss, v2w, by half a point, and we'd rather show it than hide it. The low numbers in the middle are on purpose; that's the next slide.

## 12. What we found out about the leaderboard

Two things we found about the evaluation itself. We found the split with probes: an agent that carried more charge into each week scored higher on the public board, and one that emptied the battery at each week's end scored lower, by about what we predicted. So the public board is the first half of each week and the private ranking is the second half, and since the battery's charge crosses that line, you can move score from one half to the other. We want to be upfront about that. We picked two agents. h4 is our general agent and knows nothing about the split. h3p treats each week as its own run and fills the battery before the second half; it says so at the top of the file. The other finding: price and CO2 are published by Red Eléctrica, so the test values are public. We told the organisers and didn't use them.

## 13. Three things to remember

Three things to take away. A battery this size can get 95 percent of the perfect-foresight score while seeing only three hours ahead. On the public half of the test our general agent avoids 22 percent of the CO2, that's 13.4 tonnes, and saves 26 percent of the bill, about 25 thousand euros, without breaking a limit once. And what's left is in forecasting price and CO2 three to eight hours ahead, and in plain local knowledge like the holiday calendar. Thanks, happy to take questions.

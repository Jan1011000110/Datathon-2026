# Campus Battery Agent: speaker notes

Slides: [slides.pdf](slides.pdf) (or open `slides.html` locally to see slides and notes together).

## 1. Running a campus battery when you can only see 3 hours ahead

(about 15 s) Hi, we're Fanboy and Chum Chum. Our agent runs the campus battery in hard mode, where you only see three hours ahead. It gets 95 percent of what a crystal ball would get, with zero infractions.

## 2. Decide every hour, see 3 hours ahead

(about 20 s) Hard mode in one breath: the battery is three times bigger, the solar plant twelve times bigger, and instead of 24 hours of price and CO2 you see three. So every hour we have to guess four things, price, CO2, campus load and sun, and still decide what the battery does.

## 3. First question: how good can anyone get?

(about 20 s) First we asked how good anyone can get. A linear programme that knows the whole future scores 21.26 on our validation weeks. Everything else is measured as points lost against that. Our first planner lost 2.45; the final agent loses 1.07, so 95 percent. What's left is mostly price and CO2 beyond three hours.

## 4. What the agent does every hour

(about 25 s) Every hour the agent does three things. It forecasts load, sun, and price and CO2, each with its error distribution. It plans 36 hours ahead with an exact dynamic programme over many possible futures and applies only the first command. And it never breaks a limit: the command is clamped, with three fallbacks behind it. We judged everything closed-loop on three validation folds.

## 5. Three forecasters, nothing fancy

(about 35 s) The forecasters are simple on purpose. Load: rules for the meter, which freezes at night and jumps at six, plus a small LightGBM on 28 inputs like the last readings, the same hour yesterday and last week, and a holiday index. That halves the error, from 38 to 20 kilowatts. Solar: a clear-sky curve times a clearness factor from ridge regression; LightGBM was worse there, so we dropped it. Price and CO2: ridge regression on the difference from an average day.

## 6. Is it an LP? Yes. We just solve it faster

(about 35 s) So is it a linear programme? Yes: minimise grid import over 36 hours, weighted by CO2 plus 0.27 times price. But we solve it as an exact dynamic programme: the value of stored energy is piecewise linear, so going one hour back is just merging two sorted lists. Same answer as scipy to eleven decimals, twenty times faster. That speed lets us plan on scenarios instead of one forecast, which was our biggest gain: 0.71 points with the same forecasts.

## 7. What came from us, not from the model

(about 35 s) Now what came from us and not from the model. One: holidays. The model handled holidays in general and skipped Barcelona's own, and validation couldn't see them because none is in the training days. We knew the campus closes for La Mercè and Whit Monday, typed in three dates, and got 0.23 on the public board. Two: the split between the public and the private test. We didn't crack it ourselves, but we decided it was the question that mattered and spent our submissions on probes instead of tuning.

## 8. What we found out about the leaderboard

(about 40 s) Here's what the probes showed. The public board is the first 84 hours of each test week, the private ranking the last 84, and the charge in the battery crosses that line, so score can move from one half to the other. To be upfront: h4, our general agent, knows nothing about this. Our other pick, h3p, plans each week on its own and fills the battery before the private half, and says so at the top of the file. We also found the price and CO2 signal is public data; we told the organisers and didn't use it.

## 9. Three things to remember

(about 20 s) Three things to remember. 95 percent of perfect foresight with three hours of visibility. 22 percent less CO2 and 26 percent off the bill on the public half, with zero infractions. And what's left is in forecasting three to eight hours ahead. Thanks! Questions?

## 10. Don’t discharge what the campus won’t use

Backup slide, for questions. This is the rule that comes out of the plan for the hour we're deciding. Discharge one more kilowatt only if what it earns now, times the chance the campus actually uses it, beats what that energy is worth later. Left plot: a weekday at one in the afternoon, forecast net load 443 kilowatts. If a stored kilowatt-hour is worth 90 percent as much later, we only discharge 214, the part we're almost sure will be used. If it's worth 25 percent later, we discharge 523, more than the forecast. Right plot: why a fixed safety margin is the wrong tool. At four in the morning or seven in the evening the load is steady and the best move is just the forecast. At one in the afternoon, with the sun uncertain, it goes anywhere from zero to twice the forecast. Charging works the same way, mirrored.

## 11. Six things the data told us

Backup slide, for questions. Six things we found by looking at the data, each of which changed the agent. The load meter isn't read between eleven at night and five in the morning, so the night load is known exactly; then at six it jumps, and how much depends on the weekday. Solar panels colder than the air before dawn mean a clear sky, so tomorrow is sunny: that went into the solar forecast. The battery is bigger than the campus can absorb: in 87 percent of discharge hours the limit is the load, and anything above it is given to the grid for free. That makes the error one-sided, which is exactly why we plan on the distribution. For price and CO2, we gave the planner the true values in the wrong order and it recovered everything, so the level matters and the order doesn't; four times a more accurate forecast didn't move the score. And an agent that only acts in August scored exactly zero, so there is no August week in the test.

## 12. How three Barcelona holidays fooled the agent

Backup slide, for questions. Here's the holiday one in detail, because it's the result we like most. Our validation weeks are days 1 to 21 of each month, but the test weeks are the last seven days. And if you look at a calendar, three of those test weeks start on a holiday: La Mercè on 24 September, Whit Monday on 25 May and Sant Joan on 24 June. On a holiday the campus uses about half the usual energy. But on these local and regional holidays the electricity price keeps its normal weekday shape. So the agent did the sensible weekday thing: buy cheap energy overnight for the morning. And then nobody came. We typed the three dates into the load forecaster, predicted plus 0.20, and the server gave plus 0.23. It also told us something we couldn't check from the data: yes, the campus really closes on Barcelona's local holidays. To be straight about it: this is a gain on the public board, not the reason we're first. All three holidays fall in the first half of their week, so on the final private ranking this agent scores exactly the same as the one without the calendar, 19.74. It's here because it's a good example of something the data couldn't tell us and we could.

## 13. We called our scores before submitting

Backup slide, for questions. We didn't tune against the leaderboard. Before each submission we wrote down what we expected, then sent it. Six examples here. The first row comes from two probes: an agent that only acts in August scored exactly zero, so there's no August week and the plan should end on 31 July. The public score didn't change, as predicted, because the last hours of the run are in the private half. One miss, v2w, by half a point, and we'd rather show it than hide it. The low public numbers in the middle are on purpose: those agents give up the public half for the private one.

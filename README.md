# Final-Project-Template
<!-- Edit the title above with your project title -->

## Project Overview
Topic: Can we use ML to make predictions about real estate?
This topic is important because it can help both businesses and consumers.
Project Questions:
Can ML predict a house price with different variables (regression)?
Can my mom predict house prices better than ML?
What would an answer look like:
Prediction prices in the test dataset are close to the actual price
Dot plot that shows model accuracy 
Histogram of the residuals
Bar chart of the ML prediction, mom's prediction, and the actual price
Data Sources:
Scraping Zillow
Google Maps API
Mapbox Satellite API
I'm going to merge these using address as a primary key. 


## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | ... / 10 |
| **Execution**     | ... / 20 |
| **Documentation** | ... / 10 |
| **Presentation**  | ... / 30 |
| **Total**         | ... / 70 |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
#### What did not go well?
#### What did you learn?
#### What would you do differently next time?

---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)

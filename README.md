<div align="center">
  <a href="https://github.com/EdwinJaico-Berg/Capstone-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/5277/5277239.png" alt="Logo" width="200">
  </a>

  <h3 align="center">Wildfire Predction Model</h3>

  <p align="center">
    <strong>A machine learning model that predicts the size of a wildfire given environmental features.</strong>
  </p>
  <p align="center">
    Edwin Jaico-Berg | e.j.berg@outlook.com 
  </p>
  <p align="center">
      https://github.com/EdwinJaico-Berg 
  </p>
</div>

## Motivation

2022 has presented a number of global challenges in terms of climate. This year alone stronger storms have left countries without power, while soaring summer temperatures have reduced the water level of major European rivers to the point that these major trading arteries were almost unnavigable. 

The number of wildfires has also increased staggeringly across the world given that higher temperatures and longer dry periods have created the ideal environment for the spread of catastrophically damaging fires. The Annual 2021 Wildfires Report shows that in the US alone, 7 million acres of wildland were consumed by fire that year. Consequently, annual wildfire suppression costs have risen, far exceeding the federal spending on fire prevention and hazard mitigation. 

It is with this aspect in mind that the following project has been undertaken. Using the following <a href='https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires'>wildfire data</a>, we will attempt to build a model that is able to  predict the size of wildfires based on weather and emissions (specifically greenhouse gases) data. 

# Project Structure

As can be seen, the project is split into a number of notebooks:

### <u>Preface</u>
**Loading of Emissions Data.ipynb** * <br>
*Initial loading of emissions data into easily accessible and readable files*

### <u>Data Analysis</u>
**Wildfire Data Analysis.ipynb** <br>
*Preliminary analysis of the wildfire data set* <br>
**Gather Weather Data.ipynb** * <br>
*Notebook detailing the API call made to collect the weather data* <br>
**Gathering Emissions Data.ipynb** * <br>
*Calculating the emissions data from the files created previously* <br>

### <u>EDA and Initial Models</u>
**Data Preparation, EDA, and Initial Models.ipynb** <br>
*Notebook describing EDA and initial model building process* <br>
**Creating Further Samples.ipynb** * <br>
*Getting more samples and generating relevant weather and emissions data*<br>

### <u>The Modelling Process</u>
**Regression Models.ipynb**<br>
*Building regression models including linear and decision tree regressors*<br>
**Logistic Regression.ipynb**<br>
*Building a logistic regression model*<br>
**Decision Tree.ipynb**<br>
*Building a decision tree model*<br>
**Ensemble Methods.ipynb**<br>
*Building models using AdaBoost, Gradient Boosting, and XGBoost*<br>

### <u>Changing the Classification Problem</u>
**Changing the Classification Problem.ipynb**<br>
*Creating alternate classifications for wildfires to simplify our model*

**Note:** * denotes that a notebook should not be run.
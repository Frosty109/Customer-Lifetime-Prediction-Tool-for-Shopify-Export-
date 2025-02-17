# This is the tool for the Ace Digital Customer
import csv
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor
from arviz.labels import MapLabeller

from IPython.display import Image
from pymc_marketing import clv

def main():
    print("Started...\n")
    # Set file path
    file_path = "Fourflax orders.csv"

    try: 
        data_raw = pd.read_csv(file_path)
        print("File Load Success..\n")
    except UnicodeDecodeError:
        data_raw = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Select Relevant features and calculate total sales
    features = ['Email', 'Created at', 'Payment Reference', 'Total', 'Billing Name']
    data = data_raw[features]

    # Drop rows with NaN values in the selected columns
    data = data.dropna()
    data.head()

    # Summarising the dataset's content to understand its scope
    data['Created at'] = data['Created at'].str.split(' ').str[0] # Remove the time part and keep only the date
    data['Created at'] = pd.to_datetime(data['Created at'], format="%Y-%m-%d") # Convert date to datetime format using year-month-day

    print("Date Modification Success...\n")

    # Summarising the dataset's content to understand its scope
    maxdate = data['Created at'].dt.date.max()
    mindate = data['Created at'].dt.date.min()
    unique_customers = data['Email'].nunique()
    total_sales = data['Total'].sum()

    print(f"\nThe time range of transactions is: {mindate} to {maxdate}")
    print(f"Total number of unique customers for this date range is: {unique_customers}")
    print(f"Total sales for the period is: ${total_sales}\n")

    # BG/NBD Model - RFM-T Format
    # Here, we prepare a summary from transaction data, transforming individual transaction data into data at the customer level.
    data_summary_rfm = clv.utils.clv_summary(data, 'Email', 'Created at', 'Total')
    data_summary_rfm.index = data_summary_rfm['customer_id']
    data_summary_rfm.head()

    data_summary_rfm = data_summary_rfm[data_summary_rfm['monetary_value'] > 0]

    # plotting the frequency distribution of customers
    plt.figure(figsize=(14, 10))
    plt.hist(data_summary_rfm['frequency'], bins=30, range=[0, 30], edgecolor='k', alpha=0.7)
    plt.title('Frquency Distribution')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', alpha=0.6)
    plt.savefig("exports/frequency_distribution_of_customers_plot")
    plt.close()

    one_time = round(sum(data_summary_rfm['frequency'] == 1) / float(len(data_summary_rfm)) * 100, 2)
    print(f"The percentage of customers who purchased only once: {one_time}%\n")


    """ Calculating Days Between Purchases """

    # Select distinct Customer ID and InvoiceDate
    # data['Created at'] = pd.to_datetime(data['Created at'])
    unique_purchases = data[['Email', 'Created at']].drop_duplicates()

    # Sorting values to ensure the calculation of the difference correctly
    unique_purchases = unique_purchases.sort_values(['Email', 'Created at'])

    # Calculating the difference in days between current and next purchase
    unique_purchases['NextInvoiceDate'] = unique_purchases.groupby('Email')['Created at'].shift(-1)
    unique_purchases['DaysBetween'] = (unique_purchases['NextInvoiceDate'] - unique_purchases['Created at']).dt.days

    # Calculating Average Days
    customer_avg_days = unique_purchases.groupby('Email')['DaysBetween'].mean().dropna()

    print(customer_avg_days, "\n")

    # Plotting the histogram/graph
    plt.figure(figsize=(14,10))
    plt.hist(customer_avg_days, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Average Days Between Purchases per Customer')
    plt.xlabel('Average Number of Days Between Purchases')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', alpha=0.6)
    plt.savefig("exports/average_days_between_purchases_plot")
    plt.close()

    # While the default priors work well for large datasets, they can be too broad or non-informative for smaller datasets.
    # Therefore, we're refining our model by specifying more informative priors for the model's parameters.
    # This step helps in guiding the model towards more realistic areas of the parameter space.

    model_config = {
        'a_prior': {'dist': 'HalfNormal',
                    'kwargs': {'sigma': 100}},
        'b_prior': {'dist': 'HalfNormal',
                    'kwargs': {'sigma': 100}},
        'alpha_prior': {'dist': 'HalfNormal',
                    'kwargs': {'sigma': 100}},
        'r_prior': {'dist': 'HalfNormal',
                    'kwargs': {'sigma': 100}},
    }

    # Initializing the BG/NBD model for customer lifetime value analysis.
    # This creates a Beta Geometric/Negative Binomial Distribution (BG/NBD) model using the historical RFM (Recency, Frequency, Monetary value) data.
    # The `data_summary_rfm` contains customer transaction data, and `model_config` provides configuration for the model, such as priors or hyperparameters.
    bgm = clv.BetaGeoModel(
        data=data_summary_rfm,
        model_config=model_config
    )

    # Building the BG/NBD model.
    # This step defines the probabilistic structure of the model based on the provided data and configuration.
    # It prepares the model for Bayesian inference but does not yet fit the model to the data.
    bgm.build_model()

    # Displaying the constructed model.
    # Calling `bgm` without additional methods shows a textual summary of the model, including its parameters and their prior distributions.
    bgm

    # Fitting the BG/NBD model to the data using Bayesian inference.
    # This step estimates the posterior distributions of the model's parameters by sampling from them, based on the observed customer data.
    bgm.fit()

    # Displaying a summary of the fitted model.
    # The summary includes parameter estimates (e.g., `a`, `b`, `alpha`, `r`), their standard deviations, credible intervals, and diagnostic metrics.
    # This provides insights into the reliability and stability of the parameter estimates.
    bgm.fit_summary()
    
    # Visualizing the posterior distributions of the parameters.
    # These distributions show the range of probable values for each parameter, considering the observed data.

    az.plot_posterior(bgm.fit_result)
    plt.savefig("exports/posterior_distributions_of_parameters_plot")
    plt.close()

    # Analyzing customer behavior using a Frequency/Recency matrix, which visualizes customer archetypes.
    # This matrix helps in identifying patterns like potentially "at-risk" customers who might churn.

    clv.plot_frequency_recency_matrix(bgm)
    plt.savefig("exports/frequency_recency_matrix_plot")
    plt.close()

    # The Probability Alive Matrix visualizes the likelihood of customers still being active. This view is essential for identifying potentially lost customers and understanding overall customer retention.

    clv.plot_probability_alive_matrix(bgm)
    plt.savefig("exports/probability_alive_matrix_plot")
    plt.close()

    # Predicting future purchases within the next 365 days using the fitted model. 

    future_purpose_days = 365 * 3 # * for years. Eg, 365 * 2 is 24 months or 2 years

    if future_purpose_days == 365: 
        expected_purchase_days = 12
    elif future_purpose_days > 365 and future_purpose_days <= 365 * 2:
        expected_purchase_days = 24
    elif future_purpose_days > 365 * 2 and future_purpose_days <= 365 * 3:
        expected_purchase_days = 36


    num_purchases = bgm.expected_num_purchases(
        customer_id=data_summary_rfm["customer_id"], # Unique identifier for each customer, used to match predictions to specific customers
        t=future_purpose_days, # Time horizon for prediction (365 days in this case).
        frequency=data_summary_rfm["frequency"], # The number of purchases each customer has made during the observation period.
        recency=data_summary_rfm["recency"], # The time since the customer's last purchase during the observation period.
        T=data_summary_rfm["T"] # The length of the observation period for each customer.
    )
        
    # We add expected purchases to our data summary and display the customers with the highest expected purchases

    sdata = data_summary_rfm.copy()
    sdata["expected_purchases"] = num_purchases.mean(("chain", "draw")).values
    sorted_values = sdata.sort_values(by="expected_purchases")

    print(f"\nThe expected number of purchases over {expected_purchase_days} months is:")
    print(sorted_values.tail(8), "\n")

    # Expected Number of Purchases
    # Here we find the average number of expected purchases and plot it

    average_expected_purchases_posterior = num_purchases.mean(dim="customer_id")

    az.plot_posterior(average_expected_purchases_posterior)
    plt.title(f"Average Expected Purchases Over {expected_purchase_days} Months")
    plt.savefig("exports/average_expected_number_of_purchases_plot")
    plt.close()

    # Here we export the expected number of purchases to a csv

    expected_num_purchases = sorted_values

    expected_num_purchases = expected_num_purchases.sort_values(by="expected_purchases", ascending=False)

    expected_num_purchases["expected_purchases"] = expected_num_purchases["expected_purchases"].round()
    expected_num_purchases["monetary_value"] = expected_num_purchases["monetary_value"].round(2)

    expected_num_purchases.rename(columns={"customer_id": "Email"}, inplace=True)

    expected_num_purchases.to_csv("exports/expected_num_purchases.csv", index=False)


    # Here, we're visualising the uncertainty in our predictions for the number of purchases. 
    # This step illustrates the range of likely outcomes, highlighting the probalistic nature of our predictions
    ids = sdata.sort_values(by="expected_purchases", ascending=False).head(4)['customer_id'].tolist()
    ax = az.plot_posterior(num_purchases.sel(customer_id=ids), grid=(2, 2))
    for axi, id in zip(ax.ravel(), ids):
        axi.set_title(f"Customer: {id}", size=20)
    plt.suptitle("Expected number of purchases in the next period", fontsize=28, y=1.05)
    plt.savefig("exports/expected_number_of_purchases_plot")
    plt.close()

    # Estimating the purchasing behaviour of a completely new customer.
    # This prediction is useful for understanding what purchasing behaviour might be expected from prospective customers

    az.plot_posterior(
        bgm.expected_purchases_new_customer(t=365).sel(customer_id="splupine@gmail.com")
    )
    plt.title("Expected purchases of a new customer in the first 365 days period")
    plt.savefig("exports/expected_number_of_purchases_new_cust_plot")
    plt.close()

    """ Customer Probability Histories """

    # Selecting a specific customer to visualise their transaction history and model their future transaction probability

    customer_id_viz = "koastiegirl@yahoo.com.au"
    customer_viz = data_summary_rfm.loc[customer_id_viz]
    print(f"customer_id_viz historical summary\n {customer_viz}\n") # Displaying the selected customer's historical summary

    # Constructing a hypothetical future transaction history for the selected customer over a set period(30 units of time).

    data_range = 30
    customer_viz_history = pd.DataFrame(
        dict(
            customer_id=[f"{customer_id_viz}_{i}" for i in range(data_range)],  # Include customer_id
            frequency=np.full(data_range, customer_viz["frequency"], dtype="int"),
            recency=np.full(data_range, customer_viz["recency"]),
            T=(np.arange(-1, data_range - 1) + customer_viz["recency"].astype("int"))
        )
    )

    # Calculating the probability of this customer being still active (alive) using the BG/NBD model.
    # This step predicts the likelihood of the customer returning for future purchases based on their transaction history
    
    p_alive = bgm.expected_probability_alive(data=customer_viz_history)

    print(f"Probability of being alive {p_alive}\n")

    # Visualising the customer's probability of remaining active over the forcasted period.
    # The shaded area represents the confidence interval, providing a range where the actual probability is likely to fall

    az.plot_hdi(customer_viz_history["T"], p_alive, color="C0")
    plt.plot(customer_viz_history["T"], p_alive.mean(("draw", "chain")), marker="o")
    plt.axvline(customer_viz_history["recency"].iloc[0], c="black", ls="--", label="Purchase")
    plt.title(f"Probability Customer {customer_id_viz} will purchase again")
    plt.xlabel("T")
    plt.ylabel("p")
    plt.legend()
    plt.savefig("exports/customer_repurchased_probability_plot")
    plt.close()

    # Calculating the probability average for all customers

    # Here we create a DataFrame with average data based on data_summary_rfm. This is so we can plot an average customer later

    average_customer = "Average Customer"
    average_data_rfm = {
        "customer_id": average_customer,
        "frequency": data_summary_rfm["frequency"].mean(),
        "recency": data_summary_rfm["recency"].mean(),
        "T": data_summary_rfm["T"].mean(),
        "monetary_value": data_summary_rfm["monetary_value"].mean(),
    }

    print(f"The average data summary {average_data_rfm}\n")

    # Now we constructa hypothetical future transaction history for the average customer over a set period (30 units of time).

    data_range = int(average_data_rfm["recency"] * 1.5)
    customer_avg_history = pd.DataFrame(
        dict(
            customer_id=np.arange(data_range),
            frequency=np.full(data_range, average_data_rfm["frequency"], dtype="int"),
            recency=np.full(data_range, average_data_rfm["recency"]),
            T=(np.arange(0, data_range) + average_data_rfm["recency"].astype("int"))
        )
    )

    p_alive_average = bgm.expected_probability_alive(data=customer_avg_history)

    print(f"Probability of being alive average {p_alive_average}\n")

    print("Recency value:", customer_avg_history["recency"].iloc[0])
    print("T range:", customer_avg_history["T"].min(), customer_avg_history["T"].max())

    # Plotting
    az.plot_hdi(customer_avg_history["T"], p_alive_average, color="C0")
    plt.plot(customer_avg_history["T"], p_alive_average.mean(("draw", "chain")), marker="o")

    # Adding axvline
    plt.axvline(
        customer_avg_history["recency"].iloc[0], c="black", ls="--", label="Purchase"
    )

    # Adjust x-axis to include the recency value
    plt.xlim(min(customer_avg_history["T"].min(), customer_avg_history["recency"].iloc[0]) - 1,
            max(customer_avg_history["T"].max(), customer_avg_history["recency"].iloc[0]) + 1)

    plt.title("Probability That The Average Customer Will Buy Again")
    plt.xlabel("T (time)")
    plt.ylabel("p (probability)")
    plt.legend()
    plt.show()

    return 1
    
    """ Gamma-Gamma model """
    # Filtering out customers with zero frequency to fit the Gamma-Gamma model.
    # This model requires customers to have made at least one repeat purchase.

    nonzero_data = data_summary_rfm.query("frequency>0")

    # Preparing the data for the Gamma-Gamma model. This model predicts the average transaction value for each customer, 
    # given the number of transactions they have made (frequency) and their average transaction value so far.

    dataset = pd.DataFrame({
        'customer_id': nonzero_data.customer_id,
        'monetary_value': nonzero_data["monetary_value"],
        'frequency': nonzero_data["frequency"]
    })

    # Initializing the Gamma-Gamma model with the prepared data.

    gg = clv.GammaGammaModel(
        data = dataset
    )
    print("Building gg model...\n")
    gg.build_model

    # Fitting the Gamma-Gamma model to the data.
    # Instead of full Bayesian inference which can be computationally intensive, we're using maximum a posteriori (MAP) estimation.
    # MAP provides a good point estimate faster, especially useful for large datasets or when computational resources are limited.

    gg.fit(fit_method="map")

    # After fitting, we examine the model's summary and posterior distributions to understand the estimated parameters' characteristics.

    gg.fit_summary()

    gg.fit()

    # Displaying a summary of the model's fit to understand the estimated parameters' characteristics better.
    # This summary provides insights into the convergence of the model and the reliability of its predictions.

    gg.fit_summary()

    # Visualizing the posterior distributions of the estimated parameters using ArviZ.
    # These plots help us assess the uncertainty surrounding our estimates, which is crucial for informed decision-making.

    print("Visualizing the posterior distributions...\n")
    az.plot_posterior(gg.fit_result)
    plt.text(
        0.5, 1.02,  # x, y position in axes coordinates (centered, slightly above the plot)
        "Visualizing the posterior distributions. Helps us assess the uncertainty surrounding our estimates",  # Your subtitle text
        size=12,             # Subtitle font size
        ha="center",         # Horizontal alignment
        transform=axi.transAxes  # Use axis coordinate system
    )
    plt.savefig("exports/posterior_distributions_plot")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    """ Predicting Customer Spending """

    # Calculating the conditional expected average order value for each customer.

    print("Calculating Customer Spend...\n")

    expected_spend = gg.expected_customer_spend(data=data_summary_rfm)

    print(expected_spend, "\n")

    # Summarizing the expected statistics of average order value and exporting as a csv.

    expected_average_order_value = az.summary(expected_spend, kind="stats")

    expected_average_order_value = expected_average_order_value.reset_index()

    expected_average_order_value = expected_average_order_value.sort_values(by="mean", ascending=False)

    expected_average_order_value["mean"] = expected_average_order_value["mean"].round(2)

    expected_average_order_value.rename(columns={"index": "Email"}, inplace=True)
    expected_average_order_value.rename(columns={"mean": "Expected average order value"}, inplace=True)
    expected_average_order_value.rename(columns={"sd": "Standard Deviation"}, inplace=True)

    expected_average_order_value['Email'] = expected_average_order_value['Email'].str.replace(r'^x\[(.*)\]$', r'\1', regex=True)

    expected_average_order_value.to_csv("exports/expected_average_order_value.csv", index=False)

    # Creating a forest plot to visualize the expected spend across customers
    average_order_value_fig, average_order_value_ax = plt.subplots(figsize=(14, 10))  # Create figure and axes

    az.plot_forest(
        expected_spend.isel(customer_id=(range(5))),
        combined=True,
        labeller=MapLabeller(var_name_map={"x": "customer"}),
        ax=average_order_value_ax  # Pass the axes object to the plot
    )

    # Set x-axis label
    average_order_value_ax.set_xlabel("Expected average order value")

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("exports/expected_average_order_value_plot")

    # Show the plot
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # Analyising the overall average expected spend accross all customers.
    # # This aggregate metric provides a broad view, useful for strategic planning and overall revenue forcasting

    expected_average_spend_all = az.summary(expected_spend.mean("customer_id"), kind="stats") 
    print("Expected average spend across all customers\n", expected_average_spend_all)

    # Plotting the posterior distribution of the average expected spend for all customers.
    # The vertical line represents the mean, giving a reference point against the distribution spread.

    az.plot_posterior(expected_spend.mean("customer_id"))
    plt.axvline(expected_spend.mean(), color="k", ls="--")
    plt.title("Expected average order value of all customers")
    plt.savefig("exports/expected_average_order_value_all_customers_plot")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    """ Predicting New Customer Spending """

    # Estimating the expected average spend of a hypothetical new customer. 
    # This prediction helps in budgeting and planning marketing strategies for customer acquisition

    az.plot_posterior(
        gg.expected_new_customer_spend()
    )
    plt.title("Expected average order value of a new customer")
    plt.savefig("exports/expected_average_order_new_customer_plot")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    """ Estimating CLV (Customer Lifetime Value) """

    # Estimating CLV using Discounted Cash Flow approach, factoring in the time value of money with a discount rate
    # This method captures both the frequency and monetary value dimensions of CLV, offering a naunced view of customer profitability

    print(data_summary_rfm.head())

    future_t_period = 36 # months

    clv_estimate = gg.expected_customer_lifetime_value(
        transaction_model=bgm,
        data=data_summary_rfm,
        future_t=future_t_period, # months 
        discount_rate=0.01, # monthy discount rate ~ 12.7% annually
        time_unit="D", # original data is in weeks
    )

    print("The CLV estimate", clv_estimate)

    # Summarizing the expected CLV 

    expected_clv_summary_df = az.summary(clv_estimate, kind="stats")

    print("expected_clv_summary_df\n", expected_clv_summary_df)

    # Average CLV for all customers 

    az.plot_posterior(clv_estimate.mean("customer_id"))
    plt.axvline(clv_estimate.mean(), color="k", ls="--")
    plt.title(f"Average Customer Lifetime Value Over {future_t_period} Months in ($)")
    plt.savefig("exports/average_customer_lifetime_value_plot")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    clv_average = expected_clv_summary_df["mean"].mean().round(2)
    sd_average = expected_clv_summary_df["sd"].mean().round(2)
    hdi_3_average = expected_clv_summary_df["hdi_3%"].mean().round(2)
    hdi_97_average = expected_clv_summary_df["hdi_97%"].mean().round(2)

    print(f"The average Customer Lifetime Value (CLV) over a {future_t_period} month period is ${clv_average}. This has a standard deviation (sd) of {sd_average}, which indicates that the average CLV may vary by ${sd_average}. A larger sd can indicate a more inaccurate result. hdi_3% and hdi_97% represent the 3% and 97% highest density intervals (HDI) of the CLV estimate. They provide a 94% confidence range for the CLV estimate. In this case we are 94% confident the average CLV lies between ${hdi_3_average} and ${hdi_97_average}.\n")

    # Exporting the CLV data as a CSV

    expected_clv_summary_df = expected_clv_summary_df.reset_index() # Reset the index to move the 'index' column into the DataFrame

    expected_clv_summary_df = expected_clv_summary_df.sort_values(by="mean", ascending=False)

    expected_clv_summary_df["mean"] = expected_clv_summary_df["mean"].round(2)

    expected_clv_summary_df.rename(columns={"index": "Email"}, inplace=True)
    expected_clv_summary_df.rename(columns={"mean": "Expected CLV (12 month timeframe)"}, inplace=True)
    expected_clv_summary_df.rename(columns={"sd": "Standard Deviation"}, inplace=True)

    expected_clv_summary_df['Email'] = expected_clv_summary_df['Email'].str.replace(r'^x\[(.*)\]$', r'\1', regex=True)

    expected_average_order_value['Email'] = expected_average_order_value['Email'].str.replace(r'^x\[(.*)\]$', r'\1', regex=True)

    expected_clv_summary_df.to_csv("exports/customer_lifetime_value_summary.csv", index=False)


    # Visualising the estimated CLV for a subset of customers.
    # The forest plot allows for easy comparison and highlights the variability in the projected CLVs.
    labeller=MapLabeller(var_name_map={"x": "customer"})
    az.plot_forest(clv_estimate.isel(customer_id=range(10)), combined=True, labeller=labeller)
    plt.xlabel("Expected CLV")
    plt.tight_layout()
    plt.savefig("exports/estimated_CLV_plot")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    """ Exporting and Analysing CLV Data """



if __name__ == "__main__":
    main()
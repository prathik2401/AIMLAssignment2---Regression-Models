import matplotlib.pyplot as plt

training_data = [
    {"AverageDurationOfStay": 141.13, "Departures": 4483},
    {"AverageDurationOfStay": 121.85, "Departures": 24165},
    {"AverageDurationOfStay": 144.39, "Departures": 19404},
    {"AverageDurationOfStay": 121.96, "Departures": 4522},
    {"AverageDurationOfStay": 124.18, "Departures": 13416},
    {"AverageDurationOfStay": 119.6, "Departures": 9339},
    {"AverageDurationOfStay": 131.2, "Departures": 4471},
    {"AverageDurationOfStay": 68.71, "Departures": 44484},
    {"AverageDurationOfStay": 124.46, "Departures": 19635},
    {"AverageDurationOfStay": 149.39, "Departures": 4449},
    {"AverageDurationOfStay": 169.62, "Departures": 31133},
    {"AverageDurationOfStay": 161.78, "Departures": 52847},
    {"AverageDurationOfStay": 141, "Departures": 36316},
    {"AverageDurationOfStay": 150.24, "Departures": 5253},
    {"AverageDurationOfStay": 201.49, "Departures": 9959},
    {"AverageDurationOfStay": 113.49, "Departures": 23490},
    {"AverageDurationOfStay": 161.2, "Departures": 18120},
    {"AverageDurationOfStay": 63.83, "Departures": 18660},
    {"AverageDurationOfStay": 106.11, "Departures": 34810},
    {"AverageDurationOfStay": 92.44, "Departures": 33293}
]

def predict(x):
    mean_avg_duration = sum(d["AverageDurationOfStay"] for d in training_data) / len(training_data)
    print(len(training_data))
    mean_departures = sum(d["Departures"] for d in training_data) / len(training_data)
    numerator = sum((d["AverageDurationOfStay"] - mean_avg_duration) * (d["Departures"] - mean_departures) for d in training_data)
    denominator = sum((d["AverageDurationOfStay"] - mean_avg_duration) ** 2 for d in training_data)
    m = numerator / denominator
    c = mean_departures - m * mean_avg_duration
    return m * x + c

predicted_departures = [predict(d["AverageDurationOfStay"]) for d in training_data]

plt.scatter([d["AverageDurationOfStay"] for d in training_data], [d["Departures"] for d in training_data], color='red', label='Training Data')
plt.plot([d["AverageDurationOfStay"] for d in training_data], predicted_departures, color='green', label='Linear Regression Line')
plt.xlabel('Average Duration of Stay')
plt.ylabel('Departures')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
predicted_y_intercept = predict(112)
print(predicted_y_intercept)
plt.plot([d["AverageDurationOfStay"] for d in training_data], predicted_departures, color='green', label='Linear Regression Line')
plt.axhline(y=predicted_y_intercept, color='black', linestyle='--', label=f'Predicted Y-Intercept: {predicted_y_intercept:.2f}')
plt.legend()
plt.show()

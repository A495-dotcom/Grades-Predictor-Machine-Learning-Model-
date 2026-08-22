from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


X_train = [[1], [2], [3], [4], [6], [7], [8], [9]]
y_train = [35,  45,  55,  65,  75, 80,  85,  95, 100]


model = LinearRegression()
model.fit(X_train, y_train)


hours = float(input("Enter how many hours you plan to study today: "))
predicted_score = min(100, max(0, round(model.predict([[hours]])[0], 1)))

print(f"\n📊 Predicted exam score: {predicted_score}%")
print("🎨 Visualizing grades prediction graph...")


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 6))


ax.scatter(X_train, y_train, color="#01343f", s=120, label='Past Student Data', zorder=5)


X_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax.plot(X_line, y_line, color="#3740aa", linewidth=3, label='Regression Line')

#
ax.scatter([hours], [predicted_score], color="#2D023F", s=300, marker='*', 
           label=f'Your Prediction ({hours}h ➔ {predicted_score}%)', zorder=6)


ax.set_title('Study Hours vs. Exam Score Predictor (Machine Learning Model)', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Hours Studied', fontsize=12)
ax.set_ylabel('Exam Score (%)', fontsize=12)
ax.set_xlim(0, 10)
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#444444')


plt.tight_layout()
plt.show()

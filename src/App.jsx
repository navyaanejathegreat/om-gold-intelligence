import { useEffect, useState } from "react";

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/predict")
      .then((res) => res.json())
      .then((data) => setData(data))
      .catch((err) => console.log(err));
  }, []);

  if (!data) {
    return (
      <div
        style={{
          background: "#0f1220",
          color: "white",
          height: "100vh",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          fontSize: "3rem",
        }}
      >
        Loading...
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f1220",
        color: "white",
        padding: "40px",
        textAlign: "center",
      }}
    >
      <h1
        style={{
          fontSize: "4rem",
          marginBottom: "40px",
        }}
      >
        Om Gold Intelligence
      </h1>

      <div
        style={{
          maxWidth: "800px",
          margin: "auto",
          display: "grid",
          gap: "20px",
        }}
      >
        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Current Gold Price</h2>
          <h1>₹{data.currentPrice}</h1>
        </div>

        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Tomorrow Forecast</h2>
          <h1>₹{data.predictedPrice}</h1>
        </div>

        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Predicted Change</h2>
          <h1>{data.predictedChange}%</h1>
        </div>

        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Forecast Range</h2>
          <h1>
            ₹{data.lowerBound} - ₹{data.upperBound}
          </h1>
        </div>

        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Historical Error</h2>
          <h1>±{data.mae}%</h1>
        </div>

        <div
          style={{
            background: "#1a1f35",
            padding: "25px",
            borderRadius: "15px",
          }}
        >
          <h2>Direction Accuracy</h2>
          <h1>{data.directionAccuracy}%</h1>
        </div>
      </div>
    </div>
  );
}

export default App;
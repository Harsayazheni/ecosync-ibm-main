let optimizationChart = null; // To store the chart instance

// Climate Change Prediction function
function predictClimateChange() {
    const temperature = document.getElementById("temperature").value;
    const country = document.getElementById("country").value;
    const location = document.getElementById("location").value;
    
    // Simple mockup prediction
    const result = `At ${temperature}°C in ${location}, ${country}, climate impact is predicted to be significant.`;
    
    // Display the result
    document.getElementById("prediction-result").innerText = result;
}

// Initialize the empty optimization chart when the page loads
window.onload = function() {
    const ctxOpt = document.getElementById('optimization-graph').getContext('2d');
    optimizationChart = new Chart(ctxOpt, {
        type: 'line',
        data: {
            labels: [],  // Empty labels initially
            datasets: [{
                label: 'No data yet',
                data: [],  // Empty data
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Call the function to load the electricity graph when the page loads
    loadElectricityGraph();
};

// Insights - Electricity Consumption Graph
function loadElectricityGraph() {
    const ctx = document.getElementById('electricity-graph').getContext('2d');

    const data = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [
            {
                label: 'Demanded Electricity (MW)',
                data: [3500, 4000, 4200, 3900, 4100, 4300, 4500, 4700, 4600, 4400, 4300, 4200], // Mock data
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                fill: false
            },
            {
                label: 'Generated Electricity (MW)',
                data: [3200, 3700, 4100, 3800, 4000, 4200, 4300, 4600, 4500, 4300, 4100, 4000], // Mock data
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                fill: false
            }
        ]
    };

    const options = {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    new Chart(ctx, {
        type: 'line',
        data: data,
        options: options
    });
}

// Optimization Function (runs when "Optimize" is clicked)
function optimizeTemperature() {
    const optTemp = document.getElementById("opt-temperature").value;

    // Mockup graph data
    const data = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
            label: `Temperature Optimization at ${optTemp}°C`,
            data: Array.from({length: 12}, () => Math.floor(Math.random() * 100)), // Random mockup data
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };

    // Update the chart with new data
    optimizationChart.data = data;
    optimizationChart.update();
}

function runMapReduce() {
    const resultsDiv = document.getElementById('mapreduce-results');
    resultsDiv.innerHTML = 'Running analysis...';
    
    fetch('/run-mapreduce', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data.results);
        } else {
            resultsDiv.innerHTML = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        resultsDiv.innerHTML = `Error: ${error}`;
    });
}

function displayResults(results) {
    const resultsDiv = document.getElementById('mapreduce-results');
    resultsDiv.innerHTML = `
        <h2>Analysis Results</h2>
        <div class="metrics">
            <div>Accuracy: ${(results.accuracy * 100).toFixed(2)}%</div>
            <div>True Positives: ${results.true_positives}</div>
            <div>False Positives: ${results.false_positives}</div>
            <div>True Negatives: ${results.true_negatives}</div>
            <div>False Negatives: ${results.false_negatives}</div>
        </div>
    `;
}
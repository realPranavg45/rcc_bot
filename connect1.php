<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "rcc_chatbot";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
echo "Connected successfully<br>";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Log the incoming POST data for debugging
    error_log('Received POST data: ' . print_r($_POST, true));

    $commands = $_POST['commands'];
    $queries = $_POST['queries'];
    $solutions = $_POST['solutions'];

    // Log the POST data
    echo "Commands: " . $commands . "<br>";
    echo "Queries: " . $queries . "<br>";
    echo "Solutions: " . $solutions . "<br>";

    // Split the concatenated queries and solutions
    $queriesArray = array_filter(array_map('trim', explode('|', $queries)));
    $solutionsArray = array_filter(array_map('trim', explode('|', $solutions)));

    if (count($queriesArray) !== count($solutionsArray)) {
        echo "<script>alert('The number of queries does not match the number of solutions.');</script>";
    } else {
        // Proceed with insertion
        $sql = "INSERT INTO dataset (category_id, patterns, responses, y_tags) VALUES (?, ?, ?, ?)";

        foreach ($queriesArray as $index => $query) {
            $solution = $solutionsArray[$index];

            if (!empty($query) && !empty($solution)) {
                if ($stmt = $conn->prepare($sql)) {
                    $category_id = 2;
                    $stmt->bind_param("isss", $category_id, $query, $solution, $commands);

                    if ($stmt->execute()) {
                        echo "<script>alert('New record created successfully');</script>";
                    } else {
                        error_log("Error executing statement: " . $stmt->error);
                        echo "<script>alert('Error: " . $stmt->error . "');</script>";
                    }

                    $stmt->close();
                } else {
                    error_log("Error preparing statement: " . $conn->error);
                    echo "Error: " . $conn->error;
                }
            }
        }
    }
}

$conn->close();
?>

// Global variables for elements and state
const chatInput = document.querySelector(".chat_input textarea");
const sendChatBtn = document.querySelector(".chat_input span#send_btn");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot_toggler");
const chatbotCloseBtn = document.querySelector(".close_btn");
const radioButtons = document.querySelectorAll(".chat-options input[type='radio']");
const chatOptionsDiv = document.querySelector(".chat-options");
let userMessage;
const inputInitHeight = chatInput.scrollHeight;
let optionSelected = false;

// Responses for closing messages
const closingResponses = [
    "See you later, thanks for visiting",
    "Have a nice day",
    "Bye! Come back again soon.",
];

// Function to create a chat message <li> element
const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent =
        className === "outgoing"
            ? `<p>${message}</p>`
            : `<span class="icon"></span><p></p>`;
    chatLi.innerHTML = chatContent;
    return chatLi;
};

// Function to handle generating a response from the server
const generateResponse = (message) => {
    const API_URL = `${SCRIPT_ROOT}/predict`;
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
    };

    return fetch(API_URL, requestOptions)
        .then((response) => response.json())
        .then((data) => {
            const isClosingResponse = closingResponses.includes(data.answer);
            return { response: data.answer, isClosingResponse };
        })
        .catch((error) => {
            console.error("Error:", error);
            return {
                response: "Sorry, something went wrong. Please try again later.",
                isClosingResponse: false,
            };
        });
};

// Function to simulate typing animation for chat messages
const typeMessage = (message, chatLi) => {
    const messageElem = chatLi.querySelector("p");
    let index = 0;
    const typingSpeed = 50;
    const intervalId = setInterval(() => {
        if (index < message.length) {
            messageElem.textContent += message.charAt(index);
            index++;
            chatbox.scrollTo(0, chatbox.scrollHeight);
        } else {
            clearInterval(intervalId);
        }
    }, typingSpeed);
};

// Function to handle user input and chat flow
const handleChat = () => {
    userMessage = chatInput.value.trim();
    if (!userMessage) return;
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    const outgoingLi = createChatLi(userMessage, "outgoing");
    chatbox.appendChild(outgoingLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    chatOptionsDiv.style.display = "none";

    setTimeout(() => {
        const thinkingLi = createChatLi("Thinking...", "incoming");
        chatbox.appendChild(thinkingLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);

        generateResponse(userMessage).then(({ response, isClosingResponse }) => {
            chatbox.removeChild(thinkingLi);
            const incomingLi = createChatLi("", "incoming");
            chatbox.appendChild(incomingLi);
            typeMessage(response, incomingLi);
            chatbox.scrollTo(0, chatbox.scrollHeight);

            if (isClosingResponse) {
                setTimeout(() => {
                    chatOptionsDiv.style.display = "block";
                    const chatOptionsContainer = document.createElement("div");
                    chatOptionsContainer.classList.add("chat", "incoming");
                    chatOptionsContainer.appendChild(chatOptionsDiv);
                    chatbox.appendChild(chatOptionsContainer);
                    chatbox.scrollTo(0, chatbox.scrollHeight);
                }, 2000);
            }
        });
    }, 600);
};

// Function to check if an option is selected
const isOptionSelected = () => {
    for (let radioButton of radioButtons) {
        if (radioButton.checked) {
            return true;
        }
    }
    return false;
};

// Event listeners for input resizing and message handling
chatInput.addEventListener("input", () => {
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 700) {
        e.preventDefault();
        if (isOptionSelected()) {
            handleChat();
        }
    }
});

sendChatBtn.addEventListener("click", () => {
    if (isOptionSelected()) {
        handleChat();
    }
});

// Event listeners for chatbot toggler and options
chatbotCloseBtn.addEventListener("click", () =>
    document.body.classList.remove("show_chatbot")
);
chatbotToggler.addEventListener("click", () =>
    document.body.classList.toggle("show_chatbot")
);

radioButtons.forEach((radioButton) => {
    radioButton.addEventListener("change", () => {
        optionSelected = true;
    });
});

// Document ready function for additional functionality
document.addEventListener("DOMContentLoaded", function () {
    const exitButton = document.querySelector(".circle-btn");
    if (exitButton) {
        exitButton.addEventListener("click", exitPrompt);
    }
});

// Functions related to handling the form prompt and submission
function showPrompt() {
    const promptContainer = document.querySelector(".prompt-container");
    promptContainer.classList.add("active");
    const addButton = document.querySelector(".btn-container .btn");
    if (addButton) {
        addButton.style.display = "none";
    }
}

function showForm() {
    const application = document.getElementById("application")
        ? document.getElementById("application").value
        : "";
    const formContainer = document.querySelector(".form-container");

    if (application === "") {
        alert("Please choose an application.");
        return;
    }

    let formContent = "";
    if (application === "linux") {
        formContent = `
            <label for="commands">Command:</label>
            <textarea id="commands" name="commands" placeholder="Enter command"></textarea>
            <label for="queries">Probable Queries:</label>
            <div id="query-container">
                <div class="query-box">
                    <textarea name="queries" placeholder="Enter probable query"></textarea>
                    <button type="button" class="dark-red-btn add-query-btn" onclick="addQuery()">Add Query</button>
                </div>
            </div>
            <label for="solutions">Probable Solutions:</label>
            <div id="solution-container">
                <div class="solution-box">
                    <textarea name="solutions" placeholder="Enter probable solution"></textarea>
                    <button type="button" class="dark-red-btn add-solution-btn" onclick="addSolution()">Add Solution</button>
                </div>
            </div>
        `;
    } else {
        formContent = `
            <label for="queries">Probable Query:</label>
            <div id="query-container">
                <div class="query-box">
                    <textarea name="queries" placeholder="Enter probable query"></textarea>
                    <button type="button" class="dark-red-btn add-query-btn" onclick="addQuery()">Add Query</button>
                </div>
            </div>
            <label for="solutions">Probable Solution:</label>
            <div id="solution-container">
                <div class="solution-box">
                    <textarea name="solutions" placeholder="Enter probable solution"></textarea>
                    <button type="button" class="dark-red-btn add-solution-btn" onclick="addSolution()">Add Solution</button>
                </div>
            </div>
        `;
    }

    formContent += `
        <div class="form-actions">
            <input class="dark-red-btn sub" type="submit" value="Submit">
            <button type="button" class="dark-red-btn clear-btn" onclick="clearForm()">Clear</button>
            <div class="circle-container">
                <button class="circle-btn" type="button" onclick="exitPrompt()">✖</button>
            </div>
        </div>
    `;

    formContainer.innerHTML = `<form action="/submit_form" method="post" onsubmit="validateForm(event)">${formContent}</form>`;
    document.querySelector(".prompt-container").classList.remove("active");
    formContainer.classList.add("active");

    const newExitButton = formContainer.querySelector(".circle-container .circle-btn");
    if (newExitButton) {
        newExitButton.addEventListener("click", exitPrompt);
    }
}

function addQuery() {
    const queryContainer = document.getElementById("query-container");
    const queryBox = document.createElement("div");
    queryBox.className = "query-box";
    queryBox.innerHTML = `
        <textarea name="queries"></textarea>
        <button type="button" class="dark-red-btn remove-query-btn" onclick="removeQuery(this)">Remove Query</button>
    `;
    queryContainer.appendChild(queryBox);
}

function removeQuery(button) {
    const queryBox = button.parentElement;
    queryBox.remove();
}

function addSolution() {
    const solutionContainer = document.getElementById("solution-container");
    const solutionBox = document.createElement("div");
    solutionBox.className = "solution-box";
    solutionBox.innerHTML = `
        <textarea name="solutions"></textarea>
        <button type="button" class="dark-red-btn remove-solution-btn" onclick="removeSolution(this)">Remove Solution</button>
    `;
    solutionContainer.appendChild(solutionBox);
}

function removeSolution(button) {
    const solutionBox = button.parentElement;
    solutionBox.remove();
}

function clearForm() {
    document.querySelectorAll(".form-container textarea").forEach((textarea) => (textarea.value = ""));
}

function goBack() {
    document.querySelector(".form-container").classList.remove("active");
    document.querySelector(".prompt-container").classList.add("active");
}

function exitPrompt() {
    const formContainer = document.querySelector(".form-container");
    formContainer.classList.remove("active");
    document.querySelector(".btn-container .btn").style.display = "block";
}


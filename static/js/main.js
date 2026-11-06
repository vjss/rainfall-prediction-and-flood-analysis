
document.addEventListener("DOMContentLoaded", function () {
    const dateInput = document.getElementById("start_date");

    if (dateInput) {
        const today = new Date();
        const localDate = new Date(
            today.getTime() - today.getTimezoneOffset() * 60000
        );

        const minDate = localDate.toISOString().split("T")[0];

        const futureDate = new Date(localDate);
        futureDate.setDate(futureDate.getDate() + 30);

        const maxDate = futureDate.toISOString().split("T")[0];

        dateInput.min = minDate;
        dateInput.max = maxDate;

        if (!dateInput.value) {
            dateInput.value = minDate;
        }
    }

    document.querySelectorAll(".navigation a").forEach(function (link) {
        if (link.href === window.location.href) {
            link.classList.add("active");
        }
    });
});

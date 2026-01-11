// Chatbot Widget Functionality
(function() {
    'use strict';

    function initChatbot() {
        const widget = document.getElementById('chatbot-widget');
        const minimizeButton = document.getElementById('chatbot-minimize');
        const widgetHeader = document.querySelector('.chatbot-widget-header');
        const minimizeIcon = document.getElementById('minimize-icon');
        const maximizeIcon = document.getElementById('maximize-icon');
        
        if (!widget || !minimizeButton || !widgetHeader || !minimizeIcon || !maximizeIcon) {
            return;
        }
        
        let isMinimized = false;

        // Toggle minimize/expand
        function toggleMinimize() {
            isMinimized = !isMinimized;
            
            if (isMinimized) {
                widget.classList.add('minimized');
                minimizeIcon.style.display = 'none';
                maximizeIcon.style.display = 'block';
                minimizeButton.setAttribute('aria-label', 'Maximize chat');
                // Prevent body scroll on mobile when minimized
                if (window.innerWidth <= 640) {
                    document.body.classList.remove('chatbot-open');
                }
            } else {
                widget.classList.remove('minimized');
                minimizeIcon.style.display = 'block';
                maximizeIcon.style.display = 'none';
                minimizeButton.setAttribute('aria-label', 'Minimize chat');
                // Allow body scroll on mobile when expanded
                if (window.innerWidth <= 640) {
                    document.body.classList.add('chatbot-open');
                }
            }
        }

        // Event listener on header - check if click is NOT on the button
        widgetHeader.addEventListener('click', function(e) {
            // Only toggle if clicking on header itself or title, not the button
            const clickedButton = e.target === minimizeButton || minimizeButton.contains(e.target) || e.target.closest('.chatbot-minimize');
            if (!clickedButton) {
                e.preventDefault();
                toggleMinimize();
            }
        });
        
        // Button also works
        minimizeButton.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleMinimize();
        });

        // On mobile, prevent body scroll when widget is open
        if (window.innerWidth <= 640) {
            document.body.classList.add('chatbot-open');
        }

        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (window.innerWidth <= 640) {
                    if (!isMinimized) {
                        document.body.classList.add('chatbot-open');
                    }
                } else {
                    document.body.classList.remove('chatbot-open');
                }
            }, 100);
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initChatbot);
    } else {
        // DOM already loaded, run immediately
        setTimeout(initChatbot, 0);
    }
})();

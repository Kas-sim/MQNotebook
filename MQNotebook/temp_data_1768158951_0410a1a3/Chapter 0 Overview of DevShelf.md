
# Tutorial: DevShelfGUI

DevShelfGUI is a **digital library search engine** that helps users *discover and read programming books*.
It offers an intuitive interface (both command-line and graphical) to **search for books**, provides *intelligent recommendations*,
and allows users to *view book details and read PDFs*. The system also tracks user interactions to improve recommendations.


## Visual Overview

```mermaid
flowchart TD
    A0["Book (Domain Model)
"]
    A1["Search Index Management
"]
    A2["Core Search Engine
"]
    A3["Search Enhancement & Recommendations
"]
    A4["User Interaction & Analytics
"]
    A5["Application Startup & Flow Control
"]
    A6["User Interface Presentation
"]
    A7["Text Normalization Utilities
"]
    A5 -- "Loads Books" --> A0
    A5 -- "Loads Index" --> A1
    A5 -- "Loads Text Processors" --> A7
    A5 -- "Initializes Search" --> A2
    A5 -- "Initializes Enhancements" --> A3
    A5 -- "Initializes Analytics" --> A4
    A5 -- "Initializes UI" --> A6
    A1 -- "Indexes Books" --> A0
    A1 -- "Uses for Indexing" --> A7
    A2 -- "Uses Index Data" --> A1
    A2 -- "Processes Query" --> A7
    A2 -- "Delegates Refinement" --> A3
    A3 -- "Uses Book Data" --> A0
    A3 -- "Uses Popularity" --> A4
    A3 -- "Normalizes for Suggestions" --> A7
    A4 -- "Provides Popularity Data" --> A3
    A4 -- "Records Clicks" --> A0
    A6 -- "Requests Services" --> A5
    A6 -- "Displays Books" --> A0
```

## Chapters

1. [Application Startup & Flow Control
](01_application_startup___flow_control_.md)
2. [Book (Domain Model)
](02_book__domain_model__.md)
3. [User Interface Presentation
](03_user_interface_presentation_.md)
4. [Search Index Management
](04_search_index_management_.md)
5. [Core Search Engine
](05_core_search_engine_.md)
6. [User Interaction & Analytics
](06_user_interaction___analytics_.md)
7. [Search Enhancement & Recommendations
](07_search_enhancement___recommendations_.md)
8. [Text Normalization Utilities
](08_text_normalization_utilities_.md)

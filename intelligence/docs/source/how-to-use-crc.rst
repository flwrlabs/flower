########################################
 Use Flower Confidential Remote Compute
########################################

To use Flower Confidential Remote Compute you must first subscribe from the `Flower
Intelligence page <https://flower.ai/intelligence>`_.

*********************
 Obtain your API key
*********************

Once you are connected and have a valid subscription, you can navigate to the `Projects
page <https://flower.ai/intelligence/projects>`_.

On this page, you can manage your compute projects, API keys, and secure access to the
Flower Intelligence platform. The following steps guide you through the complete setup
process:

1. **Access the Projects Dashboard**

   After logging in, you’ll land on the **Projects Dashboard**, which lists all your
   existing projects. If this is your first time using the service, the list will be
   empty.

   .. image:: /_static/signup/projects-dashboard.png
       :alt: Projects dashboard

2. **Create a New Project**

   Click on **New Project** to create your first compute project. Each project
   represents a logical environment where you can manage API keys and access compute
   resources.

   .. image:: /_static/signup/project-creation.png
       :alt: Projects creation

   Enter a descriptive name for your project (e.g., “Mobile App” or “Research
   Deployment”) and click **Create**.

   .. image:: /_static/signup/projects.png
       :alt: Projects page

3. **View Your Project Dashboard**

   The project dashboard provides an overview of your project, including API key
   management, usage statistics, and configuration options.

   .. image:: /_static/signup/project-dashboard.png
       :alt: Project dashboard

4. **Create a Management API Key**

   Before you can generate regular API keys, you must create a **Management API Key**.
   This special key allows you to manage and rotate other keys. Click **Management API
   Key** and follow the instructions.

   .. image:: /_static/signup/mgmt-key-creation.png
       :alt: Management API key creation

5. **Save the Management Key**

   Once the key is created, copy it and store it securely. **Important:** This is the
   only time you’ll be able to view the full key value. If lost, you will need to create
   a new one.

   .. image:: /_static/signup/mgmt-key-created.png
       :alt: Management API key created

6. **Access the API Keys Dashboard**

   With a management key in place, you can now create standard API keys for your
   applications. Navigate to the **API Keys** section.

   .. image:: /_static/signup/api-keys-dashboard.png
       :alt: API keys dashboard

7. **Create a New API Key**

   Click **+ API Key** and configure its parameters, including token limit and optional
   search limit. These limits help control usage and costs.

   .. image:: /_static/signup/api-key-creation.png
       :alt: API key creation

8. **Copy and Store Your API Key**

   After creation, copy the API key immediately. As with the management key, you won’t
   be able to view it again later. Use this key in your client applications to
   authenticate with Flower Intelligence services.

   .. image:: /_static/signup/api-key-created.png
       :alt: API key created

9. **Confirm Your Key is Active**

   Your new key will now appear in the dashboard. You can edit its metadata, revoke it,
   or create additional keys as needed.

   .. image:: /_static/signup/api-key-created-dashboard.png
       :alt: API keys dashboard with key

******************
 Use your API key
******************

Once all of this is done, you can use your API key in your application code to enable
Flower Confidential Remote Compute.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            // Access the singleton instance
            let fi = FlowerIntelligence.instance

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true
            fi.apiKey = "YOUR_API_KEY"

    .. tab-item:: Kotlin
        :sync: kotlin

        .. code-block:: kotlin

            import ai.flower.intelligence.FlowerIntelligence

            suspend fun main() {
                // Access the singleton instance
                val fi = FlowerIntelligence

                // Enable remote processing and provide your API key
                fi.remoteHandoff = true
                fi.apiKey = "YOUR_API_KEY"

                // ...
            }

If you want to get started with a more concrete use case, you should probably checkout
the `CRC example
<https://github.com/flwrlabs/flower/tree/main/intelligence/ts/examples/encrypted>`_ for
TypeScript we have available on our `GitHub repo <https://github.com/flwrlabs/flower>`_.

(function () {
  "use strict";

  function normalizeVersions(rawVersions) {
    if (!Array.isArray(rawVersions)) {
      return [];
    }
    return rawVersions
      .map((item) => {
        if (typeof item === "string" && item.trim() !== "") {
          return { name: item.trim(), url: item.trim() };
        }
        if (!item || typeof item !== "object") {
          return null;
        }

        const rawName = typeof item.name === "string" ? item.name.trim() : "";
        const rawUrl = typeof item.url === "string" ? item.url.trim() : "";
        const name = rawName || rawUrl;
        const url = rawUrl || rawName;

        if (!name || !url) {
          return null;
        }

        return { name, url };
      })
      .filter((item) => item !== null);
  }

  function getVersioningContainer() {
    return document.querySelector("[data-flwr-versioning]");
  }

  function getDocsBaseUrl(versioningContainer) {
    if (!versioningContainer) {
      return "https://flower.ai/docs/framework";
    }
    return (
      versioningContainer.dataset.docsBaseUrl || "https://flower.ai/docs/framework"
    );
  }

  function getDocsUiMetadataUrl(versioningContainer) {
    if (!versioningContainer) {
      return "https://flower.ai/docs/framework/docs-ui-metadata.json";
    }
    return (
      versioningContainer.dataset.docsUiMetadataUrl ||
      "https://flower.ai/docs/framework/docs-ui-metadata.json"
    );
  }

  function getCurrentVersion(versioningContainer) {
    if (!versioningContainer) {
      return "";
    }
    return versioningContainer.dataset.currentVersion || "";
  }

  function getCurrentVersionLabel(versioningContainer) {
    if (!versioningContainer) {
      return "";
    }
    return versioningContainer.dataset.currentVersionLabel || "";
  }

  async function checkPageExistence(versionUrl, currentLanguage, pagename, docsBaseUrl) {
    const newUrl = `${docsBaseUrl}/${versionUrl}/${currentLanguage}/${pagename}.html`;
    const fallbackUrl = `${docsBaseUrl}/${versionUrl}/${currentLanguage}/index.html#references`;

    try {
      const response = await fetch(newUrl);
      window.location.href = response.ok ? newUrl : fallbackUrl;
    } catch (error) {
      console.error("Error:", error);
      window.location.href = fallbackUrl;
    }
  }

  function bindVersionLinks(versioningContainer) {
    if (!versioningContainer) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    const links = versioningContainer.querySelectorAll("a[data-version-url]");
    links.forEach((link) => {
      if (link.dataset.boundVersionClick === "true") {
        return;
      }
      const versionUrl = link.dataset.versionUrl;
      if (!versionUrl) {
        return;
      }
      link.addEventListener("click", (event) => {
        event.preventDefault();
        checkPageExistence(versionUrl, currentLanguage, pagename, docsBaseUrl);
      });
      link.dataset.boundVersionClick = "true";
    });
  }

  function renderVersionList(versioningContainer, versions) {
    if (!versioningContainer || versions.length === 0) {
      return;
    }

    const list = versioningContainer.querySelector("[data-flwr-version-list]");
    if (!list) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    list.innerHTML = "";

    versions.forEach((item) => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = `${docsBaseUrl}/${item.url}/${currentLanguage}/${pagename}.html`;
      link.dataset.versionUrl = item.url;
      link.textContent = item.name;
      li.appendChild(link);
      list.appendChild(li);
    });

    bindVersionLinks(versioningContainer);
  }

  function renderAnnouncement(runtimeAnnouncement) {
    const announcementContent = document.getElementById("flwr-runtime-announcement");
    if (!announcementContent) {
      return;
    }

    const announcementContainer = announcementContent.closest(".announcement");
    if (!announcementContainer) {
      return;
    }

    const hasAnnouncement =
      runtimeAnnouncement &&
      runtimeAnnouncement.enabled === true &&
      typeof runtimeAnnouncement.html === "string" &&
      runtimeAnnouncement.html.trim() !== "";

    if (!hasAnnouncement) {
      announcementContent.innerHTML = "";
      announcementContainer.style.display = "none";
      return;
    }

    announcementContent.innerHTML = runtimeAnnouncement.html;
    announcementContainer.style.removeProperty("display");
  }

  function getLatestStableVersion(versions) {
    for (const version of versions) {
      if (version && version.url && version.url !== "main") {
        return version.name;
      }
    }
    return "";
  }

  function getLatestStableVersionUrl(versions) {
    for (const version of versions) {
      if (version && version.url && version.url !== "main") {
        return version.url;
      }
    }
    return "";
  }

  function removeOldVersionBanner() {
    const existing = document.getElementById("flwr-old-version-banner");
    if (existing) {
      existing.remove();
    }
  }

  function renderOldVersionBanner(versioningContainer, versions) {
    removeOldVersionBanner();

    if (!versioningContainer || versions.length === 0) {
      return;
    }

    const currentVersion = getCurrentVersion(versioningContainer);
    const currentVersionLabel = getCurrentVersionLabel(versioningContainer) || currentVersion;
    const latestStableVersion = getLatestStableVersion(versions);
    const latestStableVersionUrl = getLatestStableVersionUrl(versions);

    if (
      !currentVersion ||
      !latestStableVersion ||
      !latestStableVersionUrl ||
      currentVersion === "main" ||
      currentVersion === latestStableVersionUrl
    ) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    const banner = document.createElement("aside");
    banner.id = "flwr-old-version-banner";
    banner.className = "flwr-old-version-banner";

    const message = document.createElement("span");
    message.className = "flwr-old-version-banner__text";
    message.innerHTML = `This is documentation for an old version (<strong>${currentVersionLabel}</strong>).`;

    const action = document.createElement("button");
    action.type = "button";
    action.className = "flwr-old-version-banner__action";
    action.textContent = "Switch to stable version";
    action.addEventListener("click", () => {
      checkPageExistence(latestStableVersionUrl, currentLanguage, pagename, docsBaseUrl);
    });

    const close = document.createElement("button");
    close.type = "button";
    close.className = "flwr-old-version-banner__close";
    close.setAttribute("aria-label", "Dismiss old version warning");
    close.textContent = "×";
    close.addEventListener("click", () => {
      banner.remove();
    });

    banner.appendChild(message);
    banner.appendChild(action);
    banner.appendChild(close);

    const pageRoot = document.querySelector(".page");
    const announcement = document.querySelector(".announcement");
    if (announcement && announcement.parentNode) {
      announcement.parentNode.insertBefore(banner, announcement.nextSibling);
    } else if (pageRoot && pageRoot.parentNode) {
      pageRoot.parentNode.insertBefore(banner, pageRoot);
    }
  }

  function hideEmptyAnnouncementFallback() {
    const announcementContent = document.getElementById("flwr-runtime-announcement");
    if (!announcementContent) {
      return;
    }

    const announcementContainer = announcementContent.closest(".announcement");
    if (!announcementContainer) {
      return;
    }

    if (announcementContent.textContent.trim() === "") {
      announcementContainer.style.display = "none";
    }
  }

  async function loadDocsUiMetadata(docsUiMetadataUrl) {
    try {
      const response = await fetch(docsUiMetadataUrl, { cache: "no-store" });
      if (!response.ok) {
        return null;
      }
      return await response.json();
    } catch (error) {
      console.warn("Could not load docs UI metadata:", error);
      return null;
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const versioningContainer = getVersioningContainer();
    bindVersionLinks(versioningContainer);
    hideEmptyAnnouncementFallback();

    const docsUiMetadataUrl = getDocsUiMetadataUrl(versioningContainer);
    const docsUiMetadata = await loadDocsUiMetadata(docsUiMetadataUrl);
    if (!docsUiMetadata || typeof docsUiMetadata !== "object") {
      return;
    }

    const versions = normalizeVersions(docsUiMetadata.versions);
    renderVersionList(versioningContainer, versions);
    renderOldVersionBanner(versioningContainer, versions);
    renderAnnouncement(docsUiMetadata.announcement);
  });
})();

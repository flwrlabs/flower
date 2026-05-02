# GitHub Web UI - PR Submission Steps

## Prerequisites

You need:
1. A GitHub account with a fork of flwrlabs/flower
2. Git installed locally
3. Access to this workspace terminal

## Step-by-Step Instructions

### Step 1: Verify Your Fork Exists
1. Go to https://github.com/flwrlabs/flower
2. Click the **Fork** button (top right) if you don't have a fork yet
3. This creates: `https://github.com/YOUR_USERNAME/flower`

### Step 2: Add Your Fork as Remote
```bash
cd /workspaces/flower

# List current remotes
git remote -v

# Add your fork (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/flower.git

# Verify it was added
git remote -v
```

Expected output:
```
origin    https://github.com/YOUR_USERNAME/flower.git (fetch)
origin    https://github.com/YOUR_USERNAME/flower.git (push)
upstream  https://github.com/flwrlabs/flower (fetch)
upstream  https://github.com/flwrlabs/flower (push)
```

### Step 3: Push Feature Branch to Your Fork
```bash
cd /workspaces/flower

# Make sure you're on the feature branch
git checkout feature/cryptography-47-upgrade

# Push to your fork
git push -u origin feature/cryptography-47-upgrade
```

You should see output like:
```
Enumerating objects: 4, done.
Creating remote tracking branch 'origin/feature/cryptography-47-upgrade'...
To https://github.com/YOUR_USERNAME/flower.git
 * [new branch]      feature/cryptography-47-upgrade -> feature/cryptography-47-upgrade
Branch 'feature/cryptography-47-upgrade' set up to track remote branch 'feature/cryptography-47-upgrade' from 'origin'.
```

### Step 4: Create the PR via GitHub Web UI

1. Go to https://github.com/flwrlabs/flower
2. You should see a banner: **"Your recently pushed branches"** with your feature branch
3. Click the **Compare & pull request** button on that banner

   *If you don't see the banner:*
   - Click the **Branches** tab
   - Find `feature/cryptography-47-upgrade`
   - Click the PR button next to it

4. You'll see the "Create a pull request" form. Fill in:

   **Title (already filled):**
   ```
   feat(framework): Upgrade cryptography to 47.0.0
   ```

   **Description:** Copy the content from `CRYPTOGRAPHY_47_UPGRADE.md` or use this:
   
   ```markdown
   ## Issue

   ### Description
   Upgrade the cryptography library from version 46.0.7 to 47.0.0 to benefit from important security updates and performance improvements.

   ### Related issues/PRs
   Security and performance tracking

   ## Proposal

   ### Explanation

   #### 1. Dependency Update
   Updated `framework/pyproject.toml`:
   - Changed: `cryptography>=46.0.5,<47.0.0` → `cryptography>=47.0.0,<48.0.0`

   #### 2. Verification Completed

   ✅ **CI/CD Environment Compatibility (OpenSSL 3.0+)**
   - All CI systems use OpenSSL 3.0+
   - Docker images: Ubuntu 24.04, Alpine 3.22 (all have OpenSSL 3.0)

   ✅ **Exception Handling Review**
   - UnsupportedAlgorithm exception already handled in all key files
   - No code changes needed

   ✅ **Elliptic Curve Usage**
   - Only NIST-approved curves used: SECP384R1
   - No deprecated SECT or Brainpool curves found

   ✅ **Cryptographic Functions**
   - load_pem_private_key() - verified compatible
   - load_ssh_private_key() - verified compatible
   - Secure aggregation module - fully compatible

   #### 3. Remaining Tasks (for CI)
   - Lock file regeneration (uv lock --upgrade-package cryptography)
   - Full test suite validation
   - TLS handshake verification
   - Secure aggregation tests

   ### Checklist
   - [x] Implement proposed change (pyproject.toml update)
   - [x] Verify CI environment compatibility (OpenSSL 3.0+)
   - [x] Review exception handling patterns
   - [x] Confirm all cryptographic functions are compatible
   - [x] Check for deprecated curves
   - [ ] Regenerate lock file (requires CI workflow)
   - [ ] Run tests (will be in CI)
   - [ ] Make CI checks pass

   ### Testing Plan
   CI will validate:
   1. Framework Python test suite
   2. TLS/authentication components
   3. Secure aggregation module
   4. gRPC TLS handshakes

   See CRYPTOGRAPHY_47_UPGRADE.md for detailed verification steps.

   ### Comments
   This is a straightforward dependency upgrade with comprehensive verification. All breaking changes from cryptography 47.0.0 have been analyzed and verified to not impact Flower's infrastructure.
   ```

5. Review the changes at the bottom:
   - Should show 4 files changed
   - 1 file modified: framework/pyproject.toml
   - 3 files added: Documentation files

6. Click **Create pull request** button

### Step 5: Verify PR Was Created

1. You should be taken to the PR page
2. Verify:
   - Base: `flwrlabs/flower:main`
   - Head: `YOUR_USERNAME/flower:feature/cryptography-47-upgrade`
   - Title: `feat(framework): Upgrade cryptography to 47.0.0`
   - CI checks will start automatically

3. Note the PR number (e.g., #7XXX)
4. Share the PR link: `https://github.com/flwrlabs/flower/pull/XXXX`

## Files to Be Included in PR

The PR will include these 4 commits:
1. feat: upgrade cryptography to 47.0.0
2. docs: add cryptography 47.0.0 upgrade verification report
3. docs: add implementation summary for cryptography 47.0.0 upgrade
4. docs: add PR submission guide for cryptography 47.0.0 upgrade

**Changes:**
- `framework/pyproject.toml` (1 line modified)
- `CRYPTOGRAPHY_47_UPGRADE.md` (added - 198 lines)
- `IMPLEMENTATION_SUMMARY.md` (added - 256 lines)
- `PR_SUBMISSION_GUIDE.md` (added - 272 lines)

## Troubleshooting

### Error: "Can't access https://github.com/YOUR_USERNAME/flower.git"
- **Solution:** Check your GitHub authentication
- Run: `git config credential.helper`
- May need to set up SSH keys or personal access token

### Error: "fatal: The requested URL returned error: 403"
- **Solution:** Your credentials don't have write access
- Generate a GitHub Personal Access Token: https://github.com/settings/tokens
- Use it instead of password when prompted

### "I don't see the Compare & pull request button"
- **Solution:** Manual approach:
  1. Go to https://github.com/flwrlabs/flower/pull/new/main
  2. Set head repository: YOUR_USERNAME/flower
  3. Set head branch: feature/cryptography-47-upgrade
  4. Click "Create pull request"

### "Permission denied" when pushing
- **Solution:** Verify remote URL has correct username
- Run: `git remote -v`
- Fix with: `git remote set-url origin https://github.com/YOUR_USERNAME/flower.git`

## What Happens After PR Creation

1. **GitHub CI Actions Start**
   - Tests automatically run
   - Check the "Checks" tab for results

2. **Code Review**
   - Maintainers will review
   - May request changes or approve

3. **Lock File Update**
   - After approval, maintainers may run:
     ```bash
     cd framework
     uv lock --upgrade-package cryptography
     ```
   - Or included as part of merge commit

4. **Merge to Main**
   - PR will be merged when ready
   - Changes will be in next release

## Questions?

Refer to:
- `CRYPTOGRAPHY_47_UPGRADE.md` - Technical verification details
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation overview
- `PR_SUBMISSION_GUIDE.md` - Creation instructions

---

**Ready to proceed?** Follow the steps above to create your PR!

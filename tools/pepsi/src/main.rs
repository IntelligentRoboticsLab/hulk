use std::{env::current_dir, path::PathBuf};

use anyhow::{anyhow, bail, Context};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::generate;
use tokio::fs::read_dir;

use aliveness::{aliveness, Arguments as AlivenessArguments};
use cargo::{cargo, Arguments as CargoArguments, Command as CargoCommand};
use communication::{communication, Arguments as CommunicationArguments};
use hulk::{hulk, Arguments as HulkArguments};
use location::{location, Arguments as LocationArguments};
use logs::{logs, Arguments as LogsArguments};
use player_number::{player_number, Arguments as PlayerNumberArguments};
use post_game::{post_game, Arguments as PostGameArguments};
use power_off::{power_off, Arguments as PoweroffArguments};
use pre_game::{pre_game, Arguments as PreGameArguments};
use reboot::{reboot, Arguments as RebootArguments};
use repository::Repository;
use sdk::{sdk, Arguments as SdkArguments};
use shell::{shell, Arguments as ShellArguments};
use upload::{upload, Arguments as UploadArguments};
use wireless::{wireless, Arguments as WirelessArguments};

mod aliveness;
mod cargo;
mod communication;
mod hulk;
mod location;
mod logs;
mod parsers;
mod player_number;
mod post_game;
mod power_off;
mod pre_game;
mod reboot;
mod results;
mod sdk;
mod shell;
mod upload;
mod wireless;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let arguments = Arguments::parse();
    let repository_root = match arguments.repository_root {
        Some(repository_root) => repository_root,
        None => get_repository_root()
            .await
            .context("Failed to get repository root")?,
    };
    let repository = Repository::new(repository_root);

    match arguments.command {
        Command::Aliveness(arguments) => aliveness(arguments)
            .await
            .context("Failed to execute aliveness command")?,
        Command::Build(arguments) => cargo(arguments, &repository, CargoCommand::Build)
            .await
            .context("Failed to execute build command")?,
        Command::Check(arguments) => cargo(arguments, &repository, CargoCommand::Check)
            .await
            .context("Failed to execute check command")?,
        Command::Clippy(arguments) => cargo(arguments, &repository, CargoCommand::Clippy)
            .await
            .context("Failed to execute clippy command")?,
        Command::Communication(arguments) => communication(arguments, &repository)
            .await
            .context("Failed to execute communication command")?,
        Command::Completions { shell } => generate(
            shell,
            &mut Arguments::command(),
            "pepsi",
            &mut std::io::stdout(),
        ),
        Command::Hulk(arguments) => hulk(arguments)
            .await
            .context("Failed to execute hulk command")?,
        Command::Location(arguments) => location(arguments, &repository)
            .await
            .context("Failed to execute location command")?,
        Command::Logs(arguments) => logs(arguments)
            .await
            .context("Failed to execute logs command")?,
        Command::Playernumber(arguments) => player_number(arguments, &repository)
            .await
            .context("Failed to execute player_number command")?,
        Command::Postgame(arguments) => post_game(arguments)
            .await
            .context("Failed to execute post_game command")?,
        Command::Poweroff(arguments) => power_off(arguments)
            .await
            .context("Failed to execute power_off command")?,
        Command::Pregame(arguments) => pre_game(arguments, &repository)
            .await
            .context("Failed to execute pre_game command")?,
        Command::Reboot(arguments) => reboot(arguments)
            .await
            .context("Failed to execute reboot command")?,
        Command::Run(arguments) => cargo(arguments, &repository, CargoCommand::Run)
            .await
            .context("Failed to execute run command")?,
        Command::Sdk(arguments) => sdk(arguments, &repository)
            .await
            .context("Failed to execute sdk command")?,
        Command::Shell(arguments) => shell(arguments)
            .await
            .context("Failed to execute shell command")?,
        Command::Upload(arguments) => upload(arguments, &repository)
            .await
            .context("Failed to execute upload command")?,
        Command::Wireless(arguments) => wireless(arguments)
            .await
            .context("Failed to execute wireless command")?,
    }

    Ok(())
}

#[derive(Parser)]
#[clap(name = "pepsi")]
struct Arguments {
    /// Alternative repository root (if not given the parent of .git is used)
    #[arg(long)]
    repository_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Enable/disable aliveness on NAOs
    #[command(subcommand)]
    Aliveness(AlivenessArguments),
    /// Builds the code for a target
    Build(CargoArguments),
    /// Checks the code with cargo check
    Check(CargoArguments),
    /// Checks the code with cargo clippy
    Clippy(CargoArguments),
    /// Enable/disable communication
    #[command(subcommand)]
    Communication(CommunicationArguments),
    /// Generates shell completion files
    Completions {
        #[clap(name = "shell")]
        shell: clap_complete::shells::Shell,
    },
    /// Control the HULK service
    Hulk(HulkArguments),
    /// Control the configured location
    #[command(subcommand)]
    Location(LocationArguments),
    /// Logging on the NAO
    #[command(subcommand)]
    Logs(LogsArguments),
    /// Change player numbers of the NAOs in local configuration
    Playernumber(PlayerNumberArguments),
    /// Disable NAOs after a game (downloads logs, unsets wireless network, etc.)
    Postgame(PostGameArguments),
    /// Power NAOs off
    Poweroff(PoweroffArguments),
    /// Get NAOs ready for a game (sets player numbers, uploads, sets wireless network, etc.)
    Pregame(PreGameArguments),
    /// Reboot NAOs
    Reboot(RebootArguments),
    /// Runs the code for a target
    Run(CargoArguments),
    /// Manage the NAO SDK
    #[command(subcommand)]
    Sdk(SdkArguments),
    /// Opens a command line shell to a NAO
    Shell(ShellArguments),
    /// Upload the code to NAOs
    Upload(UploadArguments),
    /// Control wireless network on the NAO
    #[command(subcommand)]
    Wireless(WirelessArguments),
}

async fn get_repository_root() -> anyhow::Result<PathBuf> {
    let path = current_dir().context("Failed to get current directory")?;
    let ancestors = path.as_path().ancestors();
    for ancestor in ancestors {
        let mut directory = read_dir(ancestor)
            .await
            .with_context(|| format!("Failed to read directory {ancestor:?}"))?;
        while let Some(child) = directory.next_entry().await.with_context(|| {
            format!("Failed to get next directory entry while iterating {ancestor:?}")
        })? {
            if child.file_name() == ".git" {
                return Ok(child
                    .path()
                    .parent()
                    .ok_or_else(|| anyhow!("Failed to get parent of {child:?}"))?
                    .to_path_buf());
            }
        }
    }

    bail!("Failed to find .git directory")
}

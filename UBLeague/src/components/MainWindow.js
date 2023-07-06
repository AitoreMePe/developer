import React from 'react';
import UserAuthentication from './UserAuthentication';
import SearchWindow from './SearchWindow';
import WindowFilter from './WindowFilter';

class MainWindow extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      playerData: [],
      matchSchedule: [],
      standings: [],
      sponsors: [],
      clips: [],
      articles: [],
      vote: null,
    };
  }

  componentDidMount() {
    // Fetch data from server-side APIs and update state
    // This is a placeholder and should be replaced with actual API calls
  }

  render() {
    return (
      <div>
        <UserAuthentication />
        <SearchWindow />
        <WindowFilter />
        <div id="playerData">{this.state.playerData}</div>
        <div id="matchSchedule">{this.state.matchSchedule}</div>
        <div id="standings">{this.state.standings}</div>
        <div id="sponsorSpace">{this.state.sponsors}</div>
        <div id="clipsSection">{this.state.clips}</div>
        <div id="articlesSection">{this.state.articles}</div>
        <div id="voteSection">{this.state.vote}</div>
      </div>
    );
  }
}

export default MainWindow;
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
  },
  dashboardContainer: {
    width: '80%',
    margin: '0 auto',
  },
  welcomeText: {
    fontSize: '1.5rem',
    color: theme.palette.primary.main,
    marginBottom: theme.spacing(2),
  },
}));

export default useStyles;